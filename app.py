
import os
import pickle

import faiss
import streamlit as st

from sentence_transformers import SentenceTransformer
from groq import Groq


# ==================================================
# CONFIGURATION
# ==================================================

FAISS_FOLDER = "."

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

TOP_K = 4

GROQ_MODEL = "openai/gpt-oss-120b"


# ==================================================
# STREAMLIT PAGE
# ==================================================

st.set_page_config(

    page_title="Hospital Knowledge Assistant",

    page_icon="🏥",

    layout="wide"
)


# ==================================================
# TITLE
# ==================================================

st.title(
    "🏥 Hospital Knowledge Base Assistant"
)

st.write(
    "Ask questions about the hospital knowledge base "
    "using Retrieval-Augmented Generation (RAG)."
)


# ==================================================
# LOAD EMBEDDING MODEL
# ==================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        EMBEDDING_MODEL
    )


# ==================================================
# LOAD FAISS
# ==================================================

@st.cache_resource
def load_faiss_data():

    index_path = os.path.join(
        FAISS_FOLDER,
        "index.faiss"
    )

    chunks_path = os.path.join(
        FAISS_FOLDER,
        "chunks.pkl"
    )

    index = faiss.read_index(
        index_path
    )

    with open(
        chunks_path,
        "rb"
    ) as file:

        chunks = pickle.load(file)

    return index, chunks


# ==================================================
# GROQ CLIENT
# ==================================================

@st.cache_resource
def load_groq_client():

    api_key = os.getenv(
        "GROQ_API_KEY"
    )

    if not api_key:

        return None

    return Groq(
        api_key=api_key
    )


# ==================================================
# RETRIEVE DOCUMENTS
# ==================================================

def retrieve_context(
    query,
    embedding_model,
    index,
    chunks
):

    query_embedding = embedding_model.encode(

        [query],

        convert_to_numpy=True,

        normalize_embeddings=True
    )

    scores, indices = index.search(

        query_embedding.astype(
            "float32"
        ),

        TOP_K
    )

    retrieved_chunks = []

    for score, index_id in zip(
        scores[0],
        indices[0]
    ):

        if index_id == -1:

            continue

        chunk = chunks[index_id].copy()

        chunk["score"] = float(score)

        retrieved_chunks.append(
            chunk
        )

    return retrieved_chunks


# ==================================================
# GENERATE ANSWER
# ==================================================

def generate_answer(
    query,
    retrieved_chunks,
    groq_client
):

    if groq_client is None:

        return (
            "GROQ_API_KEY is not configured. "
            "Please add your Groq API key "
            "in Streamlit Secrets."
        )

    context_parts = []

    for chunk in retrieved_chunks:

        context_parts.append(

            f"""
Source: {chunk['source']}
Page: {chunk['page']}

{chunk['text']}
"""
        )

    context = "\n\n".join(
        context_parts
    )

    system_prompt = """
You are a Hospital Knowledge Base Assistant.

Your job is to answer questions using ONLY
the retrieved knowledge base context.

Rules:

1. Do not invent information.
2. Do not use outside knowledge.
3. If the answer is not present in the
   context, say that the information was
   not found in the hospital knowledge base.
4. Keep answers clear and concise.
5. For medical information, do not present
   yourself as a doctor.
6. Mention relevant source information
   when appropriate.
"""

    user_prompt = f"""
Knowledge Base Context
=====================

{context}

=====================

User Question:

{query}

Answer the user using only the
knowledge base context.
"""

    response = groq_client.chat.completions.create(

        model=GROQ_MODEL,

        messages=[

            {
                "role": "system",
                "content": system_prompt
            },

            {
                "role": "user",
                "content": user_prompt
            }

        ],

        temperature=0.2
    )

    return response.choices[0].message.content


# ==================================================
# CHECK FILES
# ==================================================

index_path = os.path.join(
    FAISS_FOLDER,
    "index.faiss"
)

chunks_path = os.path.join(
    FAISS_FOLDER,
    "chunks.pkl"
)


if not os.path.exists(index_path):

    st.error(
        "FAISS index not found. "
        "Make sure faiss_index/index.faiss "
        "exists in the repository."
    )

    st.stop()


if not os.path.exists(chunks_path):

    st.error(
        "chunks.pkl not found. "
        "Make sure faiss_index/chunks.pkl "
        "exists in the repository."
    )

    st.stop()


# ==================================================
# LOAD EVERYTHING
# ==================================================

embedding_model = load_embedding_model()

index, chunks = load_faiss_data()

groq_client = load_groq_client()


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.header("📊 Knowledge Base")

    st.write(
        f"Total chunks: **{len(chunks)}**"
    )

    st.write(
        f"Retrieval results: **{TOP_K}**"
    )

    st.write(
        f"Embedding model: "
        f"`all-MiniLM-L6-v2`"
    )

    st.divider()

    st.info(
        "This assistant retrieves relevant "
        "information from the hospital "
        "knowledge base before generating "
        "an answer."
    )


# ==================================================
# QUESTION
# ==================================================

query = st.text_input(

    "Ask a question",

    placeholder=(
        "Example: What emergency services "
        "does the hospital provide?"
    )
)


# ==================================================
# ASK BUTTON
# ==================================================

if st.button(
    "🔎 Ask",
    type="primary"
):

    if not query.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        # ------------------------------
        # RETRIEVAL
        # ------------------------------

        with st.spinner(
            "Searching knowledge base..."
        ):

            retrieved_chunks = retrieve_context(

                query,

                embedding_model,

                index,

                chunks
            )


        # ------------------------------
        # GENERATION
        # ------------------------------

        with st.spinner(
            "Generating answer..."
        ):

            answer = generate_answer(

                query,

                retrieved_chunks,

                groq_client
            )


        # ------------------------------
        # ANSWER
        # ------------------------------

        st.subheader(
            "💬 Answer"
        )

        st.write(
            answer
        )


        # ------------------------------
        # RAG CONTEXT
        # ------------------------------

        st.divider()

        st.subheader(
            "📚 Retrieved RAG Context"
        )

        st.caption(
            "These are the chunks retrieved "
            "from the hospital knowledge base."
        )


        for number, chunk in enumerate(

            retrieved_chunks,

            start=1
        ):

            with st.expander(

                f"Context {number} — "
                f"{chunk['source']} "
                f"(Page {chunk['page']})"
            ):

                st.write(
                    chunk["text"]
                )

                st.caption(

                    f"Similarity score: "
                    f"{chunk['score']:.4f}"
                )

                st.caption(

                    f"Source: "
                    f"{chunk['source']} | "
                    f"Page: {chunk['page']}"
                )
