import sys
from pathlib import Path

import streamlit as st


APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))

from multi_query_rag import (  # noqa: E402
    EMBEDDING_MODEL,
    FINAL_TOP_K,
    GENERATION_MODEL,
    MIN_SIMILARITY,
    NUM_QUERY_VARIATIONS,
    TOP_K_PER_QUERY,
    generate_answer,
    generate_query_variations,
    load_vector_store,
    multi_query_retrieve,
    validate_setup,
)


st.set_page_config(
    page_title="Multi-Query RAG Assistant",
    page_icon="🔎",
    layout="wide",
)


@st.cache_resource(
    show_spinner="Loading the FAISS vector store..."
)
def get_vector_store():
    validate_setup()
    return load_vector_store()


def show_details(
    queries: list[str],
    chunks: list[dict],
) -> None:
    with st.expander("Generated search queries"):
        for number, query in enumerate(
            queries,
            start=1,
        ):
            label = (
                "Original"
                if number == 1
                else f"Variation {number - 1}"
            )
            st.markdown(f"**{label}:** {query}")

    with st.expander("Retrieved sources"):
        rows = []

        for number, chunk in enumerate(
            chunks,
            start=1,
        ):
            rows.append(
                {
                    "#": number,
                    "File": chunk["source"],
                    "Page": chunk["page"],
                    "Chunk": chunk["chunk"],
                    "Similarity": round(
                        chunk["best_similarity"],
                        4,
                    ),
                    "RRF": round(
                        chunk["rrf_score"],
                        4,
                    ),
                    "Matched queries": len(
                        chunk["matched_queries"]
                    ),
                }
            )

        st.table(rows)


try:
    index, metadata = get_vector_store()
except Exception as error:
    st.error(
        "Could not load the vector store. Run "
        "rag2_create_embeddings.py and "
        "rag2_faiss_index.py first."
    )
    st.exception(error)
    st.stop()


document_names = sorted(
    {
        record.get("source", "Unknown")
        for record in metadata
    }
)


with st.sidebar:
    st.header("RAG configuration")
    st.metric("Stored vectors", index.ntotal)
    st.metric(
        "Source documents",
        len(document_names),
    )

    st.markdown("**Documents**")
    for name in document_names:
        st.write(f"- {name}")

    st.divider()
    st.write(f"Generation: {GENERATION_MODEL}")
    st.write(f"Embedding: {EMBEDDING_MODEL}")
    st.write(
        f"Query variations: "
        f"{NUM_QUERY_VARIATIONS}"
    )
    st.write(
        f"Results per query: "
        f"{TOP_K_PER_QUERY}"
    )
    st.write(f"Final chunks: {FINAL_TOP_K}")
    st.write(
        f"Similarity threshold: "
        f"{MIN_SIMILARITY}"
    )

    if st.button(
        "Clear conversation",
        use_container_width=True,
    ):
        st.session_state.messages = []
        st.rerun()


st.title("🔎 Multi-Query RAG Assistant")
st.caption(
    "The assistant generates alternative queries, "
    "searches FAISS, and answers from the retrieved "
    "document context."
)


if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("queries"):
            show_details(
                message["queries"],
                message["sources"],
            )


question = st.chat_input(
    "Ask a question about the documents"
)


if question:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.status(
                "Running Multi-Query RAG...",
                expanded=True,
            ) as status:
                st.write(
                    "Generating alternative queries..."
                )
                variations = (
                    generate_query_variations(
                        question
                    )
                )
                all_queries = [
                    question,
                    *variations,
                ]

                st.write("Searching FAISS...")
                chunks = multi_query_retrieve(
                    question,
                    variations,
                    index,
                    metadata,
                )

                if chunks:
                    st.write(
                        "Generating the answer..."
                    )
                    answer = generate_answer(
                        question,
                        chunks,
                    )
                else:
                    answer = (
                        "No sufficiently relevant "
                        "information was found."
                    )

                status.update(
                    label="Answer ready",
                    state="complete",
                    expanded=False,
                )

            st.markdown(answer)

            if chunks:
                show_details(all_queries, chunks)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "queries": all_queries,
                    "sources": chunks,
                }
            )

        except Exception as error:
            st.error(
                "The request failed. Check Ollama "
                "and the OpenRouter API key."
            )
            st.exception(error)
