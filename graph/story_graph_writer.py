"""
graph/story_graph_writer.py — writes the User Story -> Gherkin flow's
entities into Neo4j, following the same ontology graphrag2 established:

    RequirementDoc -[:DERIVED_FROM]-> Requirement -[:REFINES]-> UserStory
    UserStory -[:ASSOCIATED_WITH]-> Persona
    UserStory -[:VALIDATES]-> AcceptanceCriteria
    UserStory -[:REQUIRES]-> Feature -[:HAS_FLOW]-> FlowStep
    TestCase -[:TESTS]-> Feature
    TestCase -[:VALIDATES]-> AcceptanceCriteria
    TestCase -[:COMPOSED_OF]-> TestStep

Coexists in the SAME Neo4j database as the main pipeline's generic
:Entity graph (graph/neo4j_manager.py's insert_graph) without
colliding — these nodes use their own labels (:UserStory, :Requirement,
...) and their own id properties (storyId, reqId, ...), never :Entity.

Per this flow's scoping decision, vector search stays on ChromaDB
(vectorstore/chroma_manager.py's user_stories collection) rather than
a Neo4j-native vector index — so, unlike graphrag2's graph_writer.py,
nothing here computes or stores an `embedding` property on any node.
Embedding a UserStory's summary and storing it in Chroma is a separate
step the call site (app.py) does right after upsert_user_story(),
keyed by the same story_id.
"""

from graph.neo4j_manager import run_write_query

CONSTRAINTS = [
    "CREATE CONSTRAINT story_req_doc_id IF NOT EXISTS FOR (n:RequirementDoc) REQUIRE n.docId IS UNIQUE",
    "CREATE CONSTRAINT story_requirement_id IF NOT EXISTS FOR (n:Requirement) REQUIRE n.reqId IS UNIQUE",
    "CREATE CONSTRAINT story_user_story_id IF NOT EXISTS FOR (n:UserStory) REQUIRE n.storyId IS UNIQUE",
    "CREATE CONSTRAINT story_persona_id IF NOT EXISTS FOR (n:Persona) REQUIRE n.personaId IS UNIQUE",
    "CREATE CONSTRAINT story_feature_id IF NOT EXISTS FOR (n:Feature) REQUIRE n.featureId IS UNIQUE",
    "CREATE CONSTRAINT story_flow_step_id IF NOT EXISTS FOR (n:FlowStep) REQUIRE n.stepId IS UNIQUE",
    "CREATE CONSTRAINT story_ac_id IF NOT EXISTS FOR (n:AcceptanceCriteria) REQUIRE n.acId IS UNIQUE",
    "CREATE CONSTRAINT story_test_case_id IF NOT EXISTS FOR (n:TestCase) REQUIRE n.caseId IS UNIQUE",
]


def initialize_story_schema() -> list:
    """
    Run once (idempotent — every statement is IF NOT EXISTS). Returns
    any statements that failed (e.g. Neo4j edition doesn't support a
    given constraint type) instead of raising, so a schema hiccup
    doesn't block ingestion entirely — same degrade-visibly pattern
    the rest of this project follows.
    """
    failed = []
    for stmt in CONSTRAINTS:
        try:
            run_write_query(stmt)
        except Exception as e:
            failed.append(f"{stmt}: {e}")
    return failed


def upsert_requirement_doc(doc_id: str, name: str, version: str, uploaded_at: str):
    run_write_query(
        """
        MERGE (d:RequirementDoc {docId: $doc_id})
        SET d.name = $name, d.version = $version, d.uploadedAt = $uploaded_at
        """,
        doc_id=doc_id, name=name, version=version, uploaded_at=uploaded_at,
    )


def upsert_requirement(req_id: str, title: str, description: str, source: str, doc_id: str = None):
    run_write_query(
        """
        MERGE (r:Requirement {reqId: $req_id})
        SET r.title = $title, r.description = $description, r.source = $source
        WITH r
        OPTIONAL MATCH (d:RequirementDoc {docId: $doc_id})
        FOREACH (_ IN CASE WHEN d IS NOT NULL THEN [1] ELSE [] END |
            MERGE (d)-[:DERIVED_FROM]->(r)
        )
        """,
        req_id=req_id, title=title, description=description, source=source, doc_id=doc_id,
    )


def upsert_user_story(story_id: str, title: str, summary: str,
                       requirement_ids: list = None, persona_id: str = None):
    run_write_query(
        """
        MERGE (s:UserStory {storyId: $story_id})
        SET s.title = $title, s.summary = $summary
        WITH s
        UNWIND $requirement_ids AS req_id
            MATCH (r:Requirement {reqId: req_id})
            MERGE (r)-[:REFINES]->(s)
        WITH s
        OPTIONAL MATCH (p:Persona {personaId: $persona_id})
        FOREACH (_ IN CASE WHEN p IS NOT NULL THEN [1] ELSE [] END |
            MERGE (s)-[:ASSOCIATED_WITH]->(p)
        )
        """,
        story_id=story_id, title=title, summary=summary,
        requirement_ids=requirement_ids or [], persona_id=persona_id,
    )


def upsert_persona(persona_id: str, role: str, context: str):
    run_write_query(
        "MERGE (p:Persona {personaId: $persona_id}) SET p.role = $role, p.context = $context",
        persona_id=persona_id, role=role, context=context,
    )


def upsert_acceptance_criteria(ac_id: str, text: str, ac_type: str, story_id: str = None):
    run_write_query(
        """
        MERGE (ac:AcceptanceCriteria {acId: $ac_id})
        SET ac.text = $text, ac.type = $ac_type
        WITH ac
        OPTIONAL MATCH (s:UserStory {storyId: $story_id})
        FOREACH (_ IN CASE WHEN s IS NOT NULL THEN [1] ELSE [] END |
            MERGE (s)-[:VALIDATES]->(ac)
        )
        """,
        ac_id=ac_id, text=text, ac_type=ac_type, story_id=story_id,
    )


def upsert_feature(feature_id: str, name: str, module: str, story_id: str = None):
    run_write_query(
        """
        MERGE (f:Feature {featureId: $feature_id})
        SET f.name = $name, f.module = $module
        WITH f
        OPTIONAL MATCH (s:UserStory {storyId: $story_id})
        FOREACH (_ IN CASE WHEN s IS NOT NULL THEN [1] ELSE [] END |
            MERGE (s)-[:REQUIRES]->(f)
        )
        """,
        feature_id=feature_id, name=name, module=module, story_id=story_id,
    )


def upsert_flow_steps(feature_id: str, steps: list, edges: list = None):
    """
    steps: [{"id": "S1", "action": "...", "type": "Action"|"Decision"}, ...]
    edges: [{"source": "S1", "target": "D1", "condition": "Yes"|None}, ...]
    """
    run_write_query(
        """
        MATCH (f:Feature {featureId: $feature_id})
        WITH f
        UNWIND $steps AS step
            MERGE (fs:FlowStep {stepId: step.id})
            SET fs.action = step.action, fs.type = step.type
            MERGE (f)-[:HAS_FLOW]->(fs)
        """,
        feature_id=feature_id, steps=steps,
    )
    if edges:
        run_write_query(
            """
            UNWIND $edges AS edge
                MATCH (a:FlowStep {stepId: edge.source})
                MATCH (b:FlowStep {stepId: edge.target})
                MERGE (a)-[rel:NEXT]->(b)
                SET rel.condition = edge.condition
            """,
            edges=edges,
        )


def write_test_case(case_id: str, title: str, case_type: str, created_at: str,
                     source_story_id: str = None, feature_id: str = None,
                     ac_ids: list = None, steps: list = None):
    """
    steps: [{"stepIndex": 1, "action": "...", "expectedResult": "..."}, ...]
    Links TestCase -[:TESTS]-> Feature, -[:VALIDATES]-> AcceptanceCriteria,
    -[:COMPOSED_OF]-> TestStep, and -[:GENERATED_FROM]-> the UserStory
    ("use case") that was actually typed in to generate it — this last
    edge is what makes "generate from this document" traceable end to
    end: TestCase -> UserStory -> Requirement -> RequirementDoc.
    """
    run_write_query(
        """
        MERGE (tc:TestCase {caseId: $case_id})
        SET tc.title = $title, tc.type = $case_type, tc.createdAt = $created_at
        WITH tc
        OPTIONAL MATCH (f:Feature {featureId: $feature_id})
        FOREACH (_ IN CASE WHEN f IS NOT NULL THEN [1] ELSE [] END |
            MERGE (tc)-[:TESTS]->(f)
        )
        WITH tc
        OPTIONAL MATCH (s:UserStory {storyId: $source_story_id})
        FOREACH (_ IN CASE WHEN s IS NOT NULL THEN [1] ELSE [] END |
            MERGE (tc)-[:GENERATED_FROM]->(s)
        )
        WITH tc
        UNWIND $ac_ids AS ac_id
            MATCH (ac:AcceptanceCriteria {acId: ac_id})
            MERGE (tc)-[:VALIDATES]->(ac)
        WITH tc
        UNWIND $steps AS step
            CREATE (ts:TestStep {stepIndex: step.stepIndex, action: step.action,
                                  expectedResult: step.expectedResult})
            MERGE (tc)-[:COMPOSED_OF]->(ts)
        """,
        case_id=case_id, title=title, case_type=case_type, created_at=created_at,
        source_story_id=source_story_id, feature_id=feature_id,
        ac_ids=ac_ids or [], steps=steps or [],
    )


def story_node_counts() -> dict:
    """Quick per-label counts for the sidebar/UI — mirrors what
    graphrag2's node_counts() showed, using this project's
    run_write_query() (a read-only MATCH...count() works fine through
    the same helper)."""
    labels = ["RequirementDoc", "Requirement", "UserStory", "Persona",
              "Feature", "FlowStep", "AcceptanceCriteria", "TestCase", "TestStep"]
    counts = {}
    for label in labels:
        try:
            rows = run_write_query(f"MATCH (n:{label}) RETURN count(n) AS c")
            counts[label] = rows[0]["c"] if rows else 0
        except Exception:
            counts[label] = "?"
    return counts
