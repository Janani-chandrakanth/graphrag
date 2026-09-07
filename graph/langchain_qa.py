"""
graph/langchain_qa.py — REPLACES graph/query_engine.py as the engine
behind app.py's "Search and Query" section.

Why: the old query phase (query_engine.run_graphrag_query) answered by
retrieving Chroma-stored COMMUNITY SUMMARY TEXT (pre-written prose
about a community, generated back in Phase 2/3) and optionally
supplementing it with a live graph pull. That's fundamentally
retrieval-over-summaries — the LLM never actually queries the graph
itself, it just reads a paragraph someone (an earlier LLM pass) wrote
about part of it.

This module replaces that with the pattern from the referenced
walkthrough: LangChain's GraphCypherQAChain. Given a natural-language
question, an LLM (ChatOllama — this project's existing Ollama server,
no new provider) writes the actual Cypher query, langchain-neo4j's
Neo4jGraph executes it against the live database, and a second LLM
pass turns the raw Cypher result rows into a natural-language answer.
The graph itself is the source of truth on every call — there's no
pre-written summary text sitting between the question and the graph
that can go stale.

allow_dangerous_requests=True is required by GraphCypherQAChain since
langchain 0.2 (it executes LLM-generated Cypher against a real
database). This project's Neo4j instance only ever holds data this
same pipeline wrote (graph_builder.py / neo4j_manager.py), and the
question box is a single-user local tool, not a public endpoint — the
same trust boundary the existing raw "Cypher Query" mode in app.py
already assumes (it lets you type and run ANY Cypher directly).
"""

from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_ollama import ChatOllama

from config import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, NEO4J_DATABASE, OLLAMA_URL, EXTRACTION_MODEL

_CYPHER_GENERATION_TEMPLATE = """Task: Generate a Cypher statement to query a graph database.
Instructions:
- Use only the node labels, relationship types, and properties in the schema below.
- Every entity node has the label :Entity with properties: id, name, type (e.g. 'Actor',
  'Feature', 'BusinessRule', 'DataObject', ...), source, community.
- Do not invent labels, relationship types, or properties not present in the schema.
- The question may begin with a bracketed list like
  "[Entities in this graph most semantically related to this question: n7 (\\"Leave
  Request\\"), n2 (\\"Approve/Reject\\") ...]". This list comes from embedding similarity,
  not literal text matching, so it can surface the right entity even when the person's
  wording shares no words with the stored name (e.g. "leave approval" finding
  "Approve/Reject"). PREFER matching directly by id against entities in this list --
  e.g. `WHERE n.id IN ["n7", "n2"]` -- over guessing free-text CONTAINS matches.
- If NO such bracketed list is present, or none of the listed entities fit the
  question, fall back to fuzzy case-insensitive matching:
  `toLower(n.name) CONTAINS toLower("term")`, never exact equality.
- Return only the Cypher statement, no explanation, no markdown fences.

Schema:
{schema}

Question: {question}
Cypher query:"""

_REFORMULATE_TEMPLATE = """The following question was asked against a software requirements \
knowledge graph, and the first search attempt found nothing. Rewrite the question using \
broader or alternate wording that might match how the graph actually labels things -- for \
example, swap a specific UI term for its general capability ("sign in" -> "login" or \
"authentication"), or a specific action for the feature it belongs to. Return ONLY the \
rewritten question, nothing else.

Original question: {question}

Rewritten question:"""

_QA_TEMPLATE = """You are answering a question about a software requirements knowledge \
graph, using the actual query result below (not general knowledge). If the result is \
empty, say plainly that nothing in the graph answers this, don't guess.

Question: {question}

Query result: {context}

Answer, in plain language:"""

_graph = None
_chain = None
_entity_embedding_cache = None   # {node_id: (name, type, embedding_vector)}


def _cosine_similarity(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _get_entity_embeddings() -> dict:
    """
    Lazily embeds every node's name once, caches in-process. This is
    the actual "understand meaning, not words" fix: CONTAINS-based
    matching (however fuzzy) is still literal substring overlap --
    "leave approval" and "Approve/Reject" share no text at all despite
    meaning the same thing. Embedding similarity catches that; string
    matching structurally cannot.

    Cost note: first call after a graph change re-embeds every node
    (one Ollama call per node) -- cheap for tens/hundreds of nodes,
    would need batching/a persisted Chroma collection instead of an
    in-process dict if the graph grows into the thousands. Invalidated
    by refresh_schema(), same trigger as the Cypher schema cache.
    """
    global _entity_embedding_cache
    if _entity_embedding_cache is not None:
        return _entity_embedding_cache

    from embeddings.embedding_model import generate_embedding
    from graph.neo4j_manager import get_all_nodes

    cache = {}
    for node in get_all_nodes():
        name = node.get("name")
        if not name:
            continue
        try:
            cache[node["id"]] = (name, node.get("type", ""), generate_embedding(name))
        except Exception:
            continue  # one bad embedding call shouldn't block the rest
    _entity_embedding_cache = cache
    return cache


def _semantic_entity_link(question: str, top_k: int = 8, min_similarity: float = 0.35) -> list:
    """Returns the top_k graph entities most semantically similar to the
    question, above min_similarity, as [(id, name, type, score), ...]
    sorted best-first. Empty list if nothing clears the similarity bar
    -- an empty hint is better than injecting irrelevant entities that
    could steer the Cypher generation toward the wrong node."""
    from embeddings.embedding_model import generate_embedding

    cache = _get_entity_embeddings()
    if not cache:
        return []

    try:
        q_embedding = generate_embedding(question)
    except Exception:
        return []

    scored = [
        (node_id, name, node_type, _cosine_similarity(q_embedding, emb))
        for node_id, (name, node_type, emb) in cache.items()
    ]
    scored.sort(key=lambda t: t[3], reverse=True)
    return [s for s in scored if s[3] >= min_similarity][:top_k]


def _build_entity_hint(question: str) -> str:
    """Formats _semantic_entity_link()'s output into the bracketed hint
    block _CYPHER_GENERATION_TEMPLATE looks for, prepended to the
    question sent into the chain. Returns the question unchanged if no
    entities cleared the similarity bar, so the prompt's CONTAINS
    fallback instruction applies exactly as it did before this change."""
    matches = _semantic_entity_link(question)
    if not matches:
        return question
    entity_list = ", ".join(f'{nid} ("{name}")' for nid, name, _type, _score in matches)
    return (
        f"[Entities in this graph most semantically related to this question: "
        f"{entity_list}]\n\n{question}"
    )


def _get_graph() -> Neo4jGraph:
    global _graph
    if _graph is None:
        kwargs = dict(
            url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD,
            enhanced_schema=True,  # samples actual property values, not just names —
                                   # meaningfully improves generated Cypher quality
                                   # for a schema this generic (single :Entity label).
            timeout=60,  # Neo4j AuraDB free tier goes to sleep after inactivity —
                         # allow extra time for the cold-start handshake.
        )
        # See config.py's NEO4J_DATABASE comment: langchain-neo4j defaults
        # this to the literal string "neo4j" if not passed, which raised
        # Neo.ClientError.Database.DatabaseNotFound on an Aura instance
        # whose actual database has a different name. Only override when
        # explicitly set, so instances where "neo4j" is correct (the
        # common case) are unaffected.
        if NEO4J_DATABASE:
            kwargs["database"] = NEO4J_DATABASE
        _graph = Neo4jGraph(**kwargs)
    return _graph


def _get_chain() -> GraphCypherQAChain:
    global _chain
    if _chain is None:
        llm = ChatOllama(model=EXTRACTION_MODEL if EXTRACTION_MODEL else "llama3.1:latest",
                          base_url=OLLAMA_URL, temperature=0)
        _chain = GraphCypherQAChain.from_llm(
            llm=llm,
            graph=_get_graph(),
            verbose=False,
            return_intermediate_steps=True,
            allow_dangerous_requests=True,
            cypher_prompt=_build_prompt(_CYPHER_GENERATION_TEMPLATE, ["schema", "question"]),
            # BUG FIX: was ["question", "query", "context"] -- but
            # _QA_TEMPLATE above only contains {question} and {context}
            # placeholders; "query" was never actually in the template
            # string. GraphCypherQAChain's QA step only ever feeds
            # question/context into this prompt (the generated Cypher
            # lands in intermediate_steps for ask_graph() to read below,
            # never in the qa_prompt itself), so declaring "query" as a
            # required input_variable meant this PromptTemplate was
            # always one variable short of what it demanded --
            # "Input to PromptTemplate is missing variables {'query'}"
            # on every single question, before Cypher generation or
            # Neo4j execution ever ran (which is why Neo4j/Ollama were
            # both fine and this still failed).
            qa_prompt=_build_prompt(_QA_TEMPLATE, ["question", "context"]),
            top_k=25,
        )
    return _chain


def _build_prompt(template: str, input_variables: list):
    from langchain_core.prompts import PromptTemplate
    return PromptTemplate(template=template, input_variables=input_variables)


def refresh_schema():
    """Call after the graph changes (new document indexed) so the next
    question's Cypher-generation prompt sees the current schema
    instead of a stale snapshot from before the graph grew. Also
    clears the semantic entity-link cache -- otherwise newly added
    nodes would be invisible to _semantic_entity_link() until the
    process restarts, silently undermining the whole point of this."""
    global _graph, _chain, _entity_embedding_cache
    _graph = None
    _chain = None
    _entity_embedding_cache = None


def _get_llm() -> ChatOllama:
    return ChatOllama(model=EXTRACTION_MODEL if EXTRACTION_MODEL else "llama3.1:latest",
                       base_url=OLLAMA_URL, temperature=0)


def _reformulate(question: str) -> str:
    """The 'rethink' step: when the first Cypher attempt returns nothing,
    ask the LLM to rephrase using broader/alternate wording, then retry
    once against the graph before falling back to vector search.
    Cheap, no training required -- see the conversation this was scoped
    in for why this replaces adopting Graph-R1's full RL loop wholesale."""
    llm = _get_llm()
    prompt = _REFORMULATE_TEMPLATE.format(question=question)
    try:
        response = llm.invoke(prompt)
        rewritten = getattr(response, "content", "").strip()
        return rewritten or question
    except Exception:
        return question


def _vector_fallback(question: str, n_results: int = 5) -> dict:
    """Last resort when the graph query comes back empty even after a
    reformulated retry: semantic search over the same requirements_chunks
    ChromaDB collection the ingestion pipeline already populates, so the
    person gets the closest matching source text instead of a hard
    'nothing found' -- doesn't touch Neo4j or Cypher at all."""
    from embeddings.embedding_model import generate_embedding
    from vectorstore.chroma_manager import search_chunks

    try:
        query_embedding = generate_embedding(question)
        raw = search_chunks(query_embedding, n_results=n_results)
    except Exception as e:
        return {"used": False, "answer": "", "chunks": [], "error": str(e)}

    documents = (raw.get("documents") or [[]])[0]
    if not documents:
        return {"used": False, "answer": "", "chunks": [], "error": None}

    llm = _get_llm()
    context = "\n\n".join(f"- {d}" for d in documents)
    prompt = (
        "The knowledge graph had no matching entities for this question. "
        "Answer using ONLY the source text excerpts below instead, and say "
        "plainly this came from source text search, not the graph, if you "
        "answer at all. If the excerpts don't actually answer it either, "
        "say so.\n\n"
        f"Question: {question}\n\nSource text excerpts:\n{context}\n\nAnswer:"
    )
    try:
        response = llm.invoke(prompt)
        answer = getattr(response, "content", "").strip()
    except Exception as e:
        return {"used": True, "answer": "", "chunks": documents, "error": str(e)}

    return {"used": True, "answer": answer, "chunks": documents, "error": None}


def _run_chain_once(question: str) -> dict:
    """One shot at chain.invoke -- factored out so ask_graph() can call
    it twice (original question, then reformulated) without duplicating
    the response-parsing logic.

    Prepends the semantic entity hint (see _build_entity_hint()) to the
    question before it reaches the chain, so Cypher generation sees the
    embedding-matched entities alongside the raw question text. Uses
    the ORIGINAL, unhinted question for the returned "cypher"/"result"
    bookkeeping and lets the QA step see the hinted version too --
    harmless there since it's just extra bracketed context ahead of
    the real question, not a replacement for it."""
    hinted_question = _build_entity_hint(question)
    try:
        chain = _get_chain()
        response = chain.invoke({"query": hinted_question})
    except Exception as e:
        return {"answer": "", "cypher": "", "result": [], "error": str(e)}

    steps = response.get("intermediate_steps", [])
    cypher = ""
    result_rows = []
    for step in steps:
        if isinstance(step, dict):
            if "query" in step:
                cypher = step["query"]
            if "context" in step:
                result_rows = step["context"]

    return {
        "answer": response.get("result", ""),
        "cypher": cypher,
        "result": result_rows,
        "error": None,
    }


def ask_graph(question: str) -> dict:
    """
    Returns:
        {"answer": str, "cypher": str, "result": list, "error": Optional[str],
         "source": "graph" | "graph_reformulated" | "vector_fallback",
         "reformulated_question": Optional[str]}
        Never raises: a chain failure (bad Cypher, Neo4j down, model
        error) surfaces as an "error" string with the other fields
        empty, so the UI can show it inline rather than crashing the
        whole Search and Query section.

    Three-step attempt, cheapest first (Graph-R1's "think -> retrieve ->
    rethink" loop, minus the RL training -- see conversation this was
    scoped in):
      1. Ask the graph directly, with fuzzy CONTAINS matching in the
         Cypher generation prompt.
      2. If that returned zero rows, reformulate the question (broader/
         alternate wording) and try the graph ONE more time.
      3. If still nothing, fall back to semantic vector search over the
         requirements_chunks ChromaDB collection instead of a hard
         "nothing found."
    """
    first = _run_chain_once(question)
    if first["error"] or first["result"]:
        first["source"] = "graph"
        first["reformulated_question"] = None
        return first

    reformulated_question = _reformulate(question)
    if reformulated_question.strip().lower() != question.strip().lower():
        second = _run_chain_once(reformulated_question)
        if second["error"] or second["result"]:
            second["source"] = "graph_reformulated"
            second["reformulated_question"] = reformulated_question
            return second

    fallback = _vector_fallback(question)
    if fallback["used"] and not fallback["error"]:
        return {
            "answer": fallback["answer"],
            "cypher": "",
            "result": fallback["chunks"],
            "error": None,
            "source": "vector_fallback",
            "reformulated_question": reformulated_question,
        }

    # Nothing worked -- return the original graph attempt's (empty)
    # response so the UI shows the honest "no answer" state, same as
    # before this change, rather than swallowing the fallback's error.
    first["source"] = "graph"
    first["reformulated_question"] = reformulated_question
    return first