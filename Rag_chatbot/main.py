import os
import math
import pickle
import hashlib
from pathlib import Path
from typing import List, Dict, Any

import streamlit as st

from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)

# ============================================================
# CONFIGURATION
# ============================================================

OLLAMA_BASE_URL = "http://localhost:11434"

LLM_MODEL = "qwen2.5:14b"
EMBEDDING_MODEL = "nomic-embed-text:latest"

DATA_DIR = Path("./rag_data")
INDEX_FILE = DATA_DIR / "vector_index.pkl"

DATA_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_EXTENSIONS = ["pdf", "txt", "md", "docx"]


# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="Local Ollama RAG",
    page_icon="🧠",
    layout="wide",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        padding-top: 1rem;
    }

    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
    }

    .app-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .app-subtitle {
        color: #888;
        margin-bottom: 2rem;
    }

    .source-box {
        background: rgba(128, 128, 128, 0.08);
        border-radius: 10px;
        padding: 10px 14px;
        margin-top: 10px;
        font-size: 0.9rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "index" not in st.session_state:
    st.session_state.index = None

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []


# ============================================================
# OLLAMA MODELS
# ============================================================

@st.cache_resource
def get_embeddings():
    return OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )


@st.cache_resource
def get_llm():
    return OllamaLLM(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.2,
    )


@st.cache_resource
def get_text_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )


# ============================================================
# FILE HELPERS
# ============================================================

def get_file_hash(file_bytes: bytes) -> str:
    """
    Generate a stable ID for a file.
    """
    return hashlib.sha256(file_bytes).hexdigest()


def save_uploaded_file(uploaded_file) -> Path:
    """
    Save uploaded file to a temporary/local location.
    """

    upload_dir = DATA_DIR / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    path = upload_dir / uploaded_file.name

    with open(path, "wb") as file:
        file.write(uploaded_file.getbuffer())

    return path


# ============================================================
# DOCUMENT LOADING
# ============================================================

def load_document(uploaded_file):
    """
    Load PDF, TXT, MD or DOCX using LangChain.
    """

    path = save_uploaded_file(uploaded_file)

    extension = path.suffix.lower()

    if extension == ".pdf":
        loader = PyPDFLoader(str(path))

    elif extension == ".txt":
        loader = TextLoader(
            str(path),
            encoding="utf-8",
            autodetect_encoding=True,
        )

    elif extension == ".md":
        loader = TextLoader(
            str(path),
            encoding="utf-8",
            autodetect_encoding=True,
        )

    elif extension == ".docx":
        loader = Docx2txtLoader(str(path))

    else:
        raise ValueError(
            f"Unsupported file type: {extension}"
        )

    documents = loader.load()

    for document in documents:
        document.metadata["source"] = uploaded_file.name

    return documents


# ============================================================
# SIMPLE LOCAL VECTOR STORE
# ============================================================

def load_index() -> Dict[str, Any]:
    """
    Load the local vector index from disk.
    """

    if not INDEX_FILE.exists():
        return {
            "documents": [],
            "vectors": [],
        }

    try:
        with open(INDEX_FILE, "rb") as file:
            return pickle.load(file)

    except Exception:
        return {
            "documents": [],
            "vectors": [],
        }


def save_index(index: Dict[str, Any]):
    """
    Save vector index to disk.
    """

    temp_file = INDEX_FILE.with_suffix(".tmp")

    with open(temp_file, "wb") as file:
        pickle.dump(
            index,
            file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    os.replace(temp_file, INDEX_FILE)


def cosine_similarity(
    vector_a: List[float],
    vector_b: List[float],
) -> float:
    """
    Calculate cosine similarity without NumPy/FAISS/Chroma.
    """

    if not vector_a or not vector_b:
        return 0.0

    if len(vector_a) != len(vector_b):
        return 0.0

    dot_product = 0.0
    norm_a = 0.0
    norm_b = 0.0

    for a, b in zip(vector_a, vector_b):
        dot_product += a * b
        norm_a += a * a
        norm_b += b * b

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot_product / (
        math.sqrt(norm_a) * math.sqrt(norm_b)
    )


# ============================================================
# INDEX DOCUMENTS
# ============================================================

def index_files(uploaded_files):
    """
    Read files, split them into chunks and generate
    Ollama embeddings.
    """

    if not uploaded_files:
        return 0

    splitter = get_text_splitter()
    embeddings = get_embeddings()

    index = load_index()

    existing_ids = {
        item["id"]
        for item in index["documents"]
    }

    new_documents = []
    new_vectors = []

    for uploaded_file in uploaded_files:

        try:
            file_bytes = uploaded_file.getvalue()

            file_hash = get_file_hash(file_bytes)

            # ------------------------------------------------
            # Prevent duplicate indexing
            # ------------------------------------------------

            if file_hash in existing_ids:
                continue

            documents = load_document(uploaded_file)

            chunks = splitter.split_documents(
                documents
            )

            if not chunks:
                continue

            texts = [
                chunk.page_content
                for chunk in chunks
            ]

            # ------------------------------------------------
            # Generate embeddings with Ollama
            # ------------------------------------------------

            vectors = embeddings.embed_documents(
                texts
            )

            for chunk, vector in zip(
                chunks,
                vectors,
            ):

                metadata = dict(
                    chunk.metadata
                )

                metadata["source"] = uploaded_file.name

                new_documents.append(
                    {
                        "id": file_hash,
                        "text": chunk.page_content,
                        "metadata": metadata,
                    }
                )

                new_vectors.append(vector)

        except Exception as error:

            st.error(
                f"Error processing "
                f"`{uploaded_file.name}`:\n\n"
                f"{error}"
            )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    if not new_documents:
        return 0

    index["documents"].extend(
        new_documents
    )

    index["vectors"].extend(
        new_vectors
    )

    save_index(index)

    st.session_state.index = index

    return len(new_documents)


# ============================================================
# SEARCH
# ============================================================

def search_documents(
    question: str,
    top_k: int = 5,
):
    """
    Semantic search over locally stored embeddings.
    """

    index = st.session_state.index

    if index is None:
        index = load_index()
        st.session_state.index = index

    if not index["documents"]:
        return []

    embeddings = get_embeddings()

    question_vector = embeddings.embed_query(
        question
    )

    scored_documents = []

    for document, vector in zip(
        index["documents"],
        index["vectors"],
    ):

        score = cosine_similarity(
            question_vector,
            vector,
        )

        scored_documents.append(
            (
                score,
                document,
            )
        )

    scored_documents.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    return scored_documents[:top_k]


# ============================================================
# BUILD CONTEXT
# ============================================================

def build_context(results):
    """
    Build context passed to Qwen.
    """

    if not results:
        return ""

    context = []

    for position, (score, document) in enumerate(
        results,
        start=1,
    ):

        source = document["metadata"].get(
            "source",
            "Unknown",
        )

        page = document["metadata"].get(
            "page"
        )

        page_info = ""

        if page is not None:
            page_info = f" | Page {page + 1}"

        context.append(
            f"""
[DOCUMENT {position}]
Source: {source}{page_info}
Relevance: {score:.3f}

{document["text"]}
"""
        )

    return "\n".join(context)


# ============================================================
# RAG RESPONSE
# ============================================================

def ask_rag(question: str):
    """
    Retrieve relevant chunks and ask Qwen.
    """

    results = search_documents(
        question,
        top_k=5,
    )

    if not results:
        return (
            "I don't know based on the provided "
            "documents.",
            [],
        )

    # --------------------------------------------------------
    # Ignore very weak matches
    # --------------------------------------------------------

    useful_results = [
        item
        for item in results
        if item[0] >= 0.20
    ]

    if not useful_results:
        return (
            "I couldn't find relevant information "
            "in the provided documents.",
            [],
        )

    context = build_context(
        useful_results
    )

    prompt = f"""
You are a local document question-answering assistant.

Your job is to answer the user's question using ONLY
the information contained in the CONTEXT.

IMPORTANT RULES:

1. Do not make up information.
2. Do not use outside knowledge.
3. If the context does not contain the answer, say:
   "I don't know based on the provided documents."
4. Give a clear, direct answer.
5. If useful, use bullet points.
6. Mention the relevant source document when possible.

CONTEXT
=======
{context}

QUESTION
========
{question}

ANSWER
======
"""

    llm = get_llm()

    answer = llm.invoke(prompt)

    sources = []

    for score, document in useful_results:

        source = document["metadata"].get(
            "source"
        )

        if source and source not in sources:
            sources.append(source)

    return answer.strip(), sources


# ============================================================
# RESET DATABASE
# ============================================================

def reset_database():
    """
    Delete local vector database.
    """

    if INDEX_FILE.exists():
        INDEX_FILE.unlink()

    uploads_dir = DATA_DIR / "uploads"

    if uploads_dir.exists():
        for file in uploads_dir.iterdir():

            try:
                if file.is_file():
                    file.unlink()

            except Exception:
                pass

    st.session_state.index = {
        "documents": [],
        "vectors": [],
    }

    st.session_state.indexed_files = []


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🧠 Local RAG")

    st.caption(
        "Ollama + LangChain + Streamlit"
    )

    st.divider()

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    st.subheader("🤖 Models")

    st.write(
        f"**LLM:** `{LLM_MODEL}`"
    )

    st.write(
        f"**Embeddings:** `{EMBEDDING_MODEL}`"
    )

    st.write(
        f"**Ollama:** `{OLLAMA_BASE_URL}`"
    )

    st.divider()

    # --------------------------------------------------------
    # Upload
    # --------------------------------------------------------

    st.subheader("📚 Documents")

    uploaded_files = st.file_uploader(
        "Upload documents",
        type=SUPPORTED_EXTENSIONS,
        accept_multiple_files=True,
    )

    if st.button(
        "📥 Index Documents",
        type="primary",
        use_container_width=True,
    ):

        if not uploaded_files:

            st.warning(
                "Please upload at least one document."
            )

        else:

            with st.spinner(
                "Creating embeddings with Ollama..."
            ):

                count = index_files(
                    uploaded_files
                )

            if count > 0:

                st.success(
                    f"Indexed {count} chunks."
                )

            else:

                st.info(
                    "No new documents were indexed."
                )

    st.divider()

    # --------------------------------------------------------
    # Database status
    # --------------------------------------------------------

    st.subheader("📊 Database")

    index = st.session_state.index

    if index is None:
        index = load_index()
        st.session_state.index = index

    document_count = len(
        index["documents"]
    )

    st.metric(
        "Indexed chunks",
        document_count,
    )

    if document_count > 0:
        st.success(
            "Vector index ready"
        )
    else:
        st.info(
            "No documents indexed yet."
        )

    st.divider()

    # --------------------------------------------------------
    # Reset
    # --------------------------------------------------------

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True,
    ):

        st.session_state.messages = []

        st.rerun()

    if st.button(
        "⚠️ Delete Vector Index",
        use_container_width=True,
    ):

        reset_database()

        st.success(
            "Vector index deleted."
        )

        st.rerun()


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    '<div class="app-title">🧠 Local RAG Chatbot</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="app-subtitle">'
    'Ask questions about your documents using '
    'Qwen 2.5 running locally through Ollama.'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# SYSTEM INFO
# ============================================================

with st.expander("⚙️ Configuration"):

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("**LLM**")
        st.code(LLM_MODEL)

    with col2:
        st.write("**Embedding**")
        st.code(EMBEDDING_MODEL)

    with col3:
        st.write("**Vector Store**")
        st.code("Local Pickle Index")


# ============================================================
# CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        sources = message.get(
            "sources",
            [],
        )

        if sources:

            with st.expander(
                "📚 Sources"
            ):

                for source in sources:
                    st.write(
                        f"• {source}"
                    )


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask a question about your documents..."
)


if question:

    # --------------------------------------------------------
    # User message
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    # --------------------------------------------------------
    # Assistant
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        if not INDEX_FILE.exists():

            answer = (
                "Please upload and index a document "
                "before asking questions."
            )

            sources = []

            st.info(answer)

        else:

            with st.spinner(
                "Searching documents and thinking..."
            ):

                try:

                    answer, sources = ask_rag(
                        question
                    )

                    st.markdown(answer)

                except Exception as error:

                    answer = (
                        "An error occurred:\n\n"
                        f"`{error}`"
                    )

                    sources = []

                    st.error(answer)

        # ----------------------------------------------------
        # Sources
        # ----------------------------------------------------

        if sources:

            with st.expander(
                "📚 Sources"
            ):

                for source in sources:
                    st.write(
                        f"• {source}"
                    )

    # --------------------------------------------------------
    # Save assistant message
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
        }
    )
