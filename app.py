import streamlit as st

from chunking.chunker import create_chunks
st.write("Chunker Module:", create_chunks.__module__)
from embeddings.embedding_model import generate_embedding
from parser.parser import extract_text
from vectorstore.chroma_manager import (
    get_collection_count,
    store_chunk,
    search_chunks
)

st.set_page_config(
    page_title="Requirement GraphRAG",
    layout="wide"
)

st.title(" Requirement GraphRAG")
import inspect

st.write(
    "Chunker File:",
    inspect.getfile(create_chunks)
)

# ==========================================================
# Clear Session
# ==========================================================

if st.button("🔄 Clear Session"):

    st.session_state.clear()

    st.rerun()

# ==========================================================
# File Upload
# ==========================================================

uploaded_file = st.file_uploader(
    "Upload Requirement Document",
    type=["pdf", "txt"]
)

if uploaded_file:

    st.success("File uploaded successfully")
    st.write("File Name:", uploaded_file.name)

    # ==========================================================
    # Build Index
    # ==========================================================

    if st.button("Build Index"):

        with st.spinner("Processing document..."):

            try:

                text = extract_text(uploaded_file)
                from graph.entity_extractor import extract_relationships

                relationships = extract_relationships(text)

                st.subheader("Extracted Relationships")

                for rel in relationships:

                    st.json(
                        {
                            "source": rel.source,
                            "relationship": rel.relationship,
                            "target": rel.target
                        }
                        )

                st.session_state["extracted_text"] = text

                chunks = create_chunks(text)
                st.write("DEBUG CHUNK COUNT:", len(chunks))

                for i, chunk in enumerate(chunks):
                     st.write(f"DEBUG Chunk {i+1} Length:", len(chunk))

                st.session_state["chunks"] = chunks

                progress_bar = st.progress(0)

                total_chunks = len(chunks)

                for idx, chunk in enumerate(chunks):

                    embedding = generate_embedding(chunk)

                    store_chunk(
                        chunk_id=f"{uploaded_file.name}_chunk_{idx}",
                        chunk_text=chunk,
                        embedding=embedding
                    )

                    progress_bar.progress(
                        (idx + 1) / total_chunks
                    )

                progress_bar.empty()

                st.session_state["indexing_success"] = True

            except Exception as e:

                st.error(f"Error: {str(e)}")

# ==========================================================
# Extracted Text
# ==========================================================

if "extracted_text" in st.session_state:

    st.subheader("Extracted Text Preview")

    st.text_area(
        "Preview",
        st.session_state["extracted_text"][:3000],
        height=250
    )



if "chunks" in st.session_state:

    chunks = st.session_state["chunks"]

    st.subheader("Chunk Information")

    text = st.session_state.get(
        "extracted_text",
        ""
    )

    st.write(
        f"Total Document Length: {len(text)} characters"
    )

    st.write(
        f"Total Chunks Created: {len(chunks)}"
    )

    for idx, chunk in enumerate(chunks):

        with st.expander(
            f"Chunk {idx + 1} ({len(chunk)} chars)"
        ):
            st.write(chunk)

# ==========================================================
# Index Success
# ==========================================================

if st.session_state.get("indexing_success"):

    st.success("Document Indexed Successfully!")

    st.write(
        f"Total Vectors Stored: {get_collection_count()}"
    )

# ==========================================================
# Search
# ==========================================================

st.divider()

st.header(" Requirement Search")

query = st.text_input(
    "Ask a question about the requirement document"
)

if query:

    try:

        query_embedding = generate_embedding(query)

        results = search_chunks(
            query_embedding=query_embedding,
            n_results=3
        )

        documents = results["documents"][0]
        distances = results["distances"][0]

        st.subheader("Retrieved Chunks")

        for idx, (document, distance) in enumerate(
            zip(documents, distances)
        ):

            st.write(
                f"Distance: {distance:.4f}"
            )

            with st.expander(
                f"Relevant Chunk {idx + 1}"
            ):
                st.write(document)

    except Exception as e:

        st.error(
            f"Search Error: {str(e)}"
        )