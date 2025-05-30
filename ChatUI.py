import os
import shutil
from pathlib import Path
import streamlit as st
import time
import tiktoken

from ChatUpload import (
    # load_documents,
    load_uploaded_documents,
    chunk_text,
    create_or_update_vector_db,
)
from ChatLLM import query_llm_cot, query_llm_norm
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma


st.title("Document Q&A with LLM and Chat History")
# Tabs for chat and DB explorer
tabs = st.tabs(["Chat Interface", "Explore VectorDBs"])

# ----------------------
# Session State Defaults
# ----------------------
def init_session_state():
    defaults = {
        'chat_history': [],
        'vector_db_name': None,
        'db': None,
        'model_selected_llm': 'llama3.2',
        'model_selected_embedding': 'nomic-embed-text',
        'reasoning_mode': 'Standard',
        'chunk_size': 300,
        'chunk_overlap': 125,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session_state()

# ----------------------
# Helpers: Load Vector DB, fetch DBs, clear question state
# ----------------------
DB_DIR = "vector_dbs"
os.makedirs(DB_DIR, exist_ok=True)

def load_vector_db(name: str, embedding_model: str) -> Chroma:
    """Load an existing Chroma vector DB"""
    persist_path = os.path.join(DB_DIR, f"{name}.db")
    embeddings = OllamaEmbeddings(model=embedding_model)
    return Chroma(persist_directory=persist_path, embedding_function=embeddings)

# Fetch list of existing DBs
_db_files = list(Path(DB_DIR).glob("*.db"))
_db_names = [f.stem for f in _db_files]

# ----------------------
# Sidebar Configuration
# ----------------------
st.sidebar.title("Configuration")

# LLM Selector
st.session_state.model_selected_llm = st.sidebar.selectbox(
    "LLM Model", ["llama3.2", "qwen3:4b", "gemma3:4b", "mistral", "deepseek-r1:8b", "deepseek-r1:7b", "phi4-mini:3.8b"],
    index=["llama3.2", "qwen3:4b", "gemma3:4b", "mistral", "deepseek-r1:8b", "deepseek-r1:7b", "phi4-mini:3.8b"].index(st.session_state.model_selected_llm)
)

#removing this as would need to make sure the same embedding model is used for the Q&A as the vector DB
# Embedding Selector
# st.session_state.model_selected_embedding = st.sidebar.selectbox(
#     "Embedding Model", ["nomic-embed-text", "snowflake-arctic-embed"],
#     index=["nomic-embed-text", "snowflake-arctic-embed"].index(st.session_state.model_selected_embedding)
# )

# Reasoning Style Toggle
st.session_state.reasoning_mode = st.sidebar.radio(
    "Reasoning Method", ["Standard", "Chain-of-Thought"],
    index=["Standard", "Chain-of-Thought"].index(st.session_state.reasoning_mode)
)

#chunking options
st.sidebar.markdown("### Chunking Options")
st.sidebar.slider(
    "Chunk size",
    min_value=100,
    max_value=2000,
    step=50,
    value=st.session_state.chunk_size,
    key="chunk_size"
)
st.sidebar.slider(
    "Chunk overlap",
    min_value=0,
    max_value=st.session_state.chunk_size,
    step=25,
    value=st.session_state.chunk_overlap,
    key="chunk_overlap"
)

# Refresh session button
if st.sidebar.button("🔄 Refresh Session"):
    st.session_state.chat_history = []
    st.sidebar.success("Session history cleared.")

# ----------------------
# Main Chat Interface
# ----------------------
with tabs[0]:
    # ----------------------
    # Step 1: Document Processing
    # ----------------------
    st.header("📁 Step 1: Upload and Process Documents")

    # Select existing Vector DB (main window)
    if _db_names:
        main_selected = st.selectbox(
            "Select Existing Vector DB (or create new below)", _db_names,
            index=_db_names.index(st.session_state.vector_db_name) if st.session_state.vector_db_name in _db_names else 0,
            key="main_selected_db"
        )
        if main_selected != st.session_state.vector_db_name:
            st.session_state.vector_db_name = main_selected
            st.session_state.db = load_vector_db(main_selected, st.session_state.model_selected_embedding)
    else:
        st.info("No existing vector databases found. Create one below.")

    #if not create a new one
    #folder_path = st.text_input("Enter folder path", key="folder_path")
    uploaded = st.file_uploader("Upload files", type=['pdf', 'docx', 'txt'], accept_multiple_files=True)
    new_db_name = st.text_input("Name for Vector DB", key="new_db_name")
    if st.button("Process and Create/Update DB"):
        documents = []
        # if folder_path and os.path.isdir(folder_path):
        #     with st.spinner("Loading documents from folder..."):
        #         documents = load_documents(folder_path)
        if not documents and uploaded:
            with st.spinner("Loading uploaded files..."):
                documents = load_uploaded_documents(uploaded)
        if not documents:
            st.error("No valid documents found. Provide a valid folder or upload files.")
        else:
            st.success(f"Loaded {len(documents)} documents.")
            with st.spinner("Chunking documents..."):
                chunks = chunk_text(documents,st.session_state.chunk_size, st.session_state.chunk_overlap)
            st.success(f"Created {len(chunks)} text chunks.")
            db_path = os.path.join(DB_DIR, f"{new_db_name}.db")
            with st.spinner("Creating/updating vector database..."):
                create_or_update_vector_db(
                    chunks,
                    model_selected_embedding=st.session_state.model_selected_embedding,
                    db_path=db_path,
                )
            st.success(f"Vector DB '{new_db_name}' is ready.")
            st.session_state.vector_db_name = new_db_name
            st.session_state.db = load_vector_db(new_db_name, st.session_state.model_selected_embedding)

    # ----------------------
    # Step 2: Chat with Documents
    # ----------------------


    st.header("💬 Chat with Documents")
    question = st.text_input("Ask a question...")

    if st.button("Get Answer"):
        if question and st.session_state.db:
            with st.spinner("Generating answer..."):
                start_time = time.time()

                if st.session_state.reasoning_mode == "Chain-of-Thought":
                    result = query_llm_cot(
                        question,
                        st.session_state.db,
                        st.session_state.model_selected_llm,
                    )
                else:
                    result = query_llm_norm(
                        question,
                        st.session_state.db,
                        st.session_state.model_selected_llm,
                    )
            st.session_state.chat_history.append({
                "question": question,
                "answer": result["answer"],
                "sources": result["sources"],
            })

            elapsed_time = time.time() - start_time
            minutes = int(elapsed_time // 60)
            seconds = int(elapsed_time % 60)
            # get the encoding
            try:
                encoding = tiktoken.encoding_for_model(st.session_state.model_selected_llm)
            except KeyError:
                # fallback if the exact model isn’t in tiktoken’s mapping
                encoding = tiktoken.get_encoding("cl100k_base")
            # count the output tokens
            token_count = len(encoding.encode(result["answer"]))
            # compute rate
            tokens_per_sec = token_count / elapsed_time if elapsed_time > 0 else 0


        # Display chat history
        for entry in reversed(st.session_state.chat_history):
            st.markdown(f"**Q:** {entry['question']}")
            st.markdown(f"**A:** {entry['answer']}")
            if entry.get('sources'):
                st.markdown("**Sources:**")
                for src in set(entry['sources']):
                    st.markdown(f"- {src}")
            st.caption(f"⏱️ Response time: {minutes} min {seconds} sec")  
            st.caption(f"🔢 Tokens: {token_count}   ⚡ {tokens_per_sec:.2f} tokens/sec")
            st.caption(f"Model Used: {st.session_state.model_selected_llm}")
            st.caption(f"Embedding Model Used: {st.session_state.model_selected_embedding}")
            st.caption(f"Database Used: {st.session_state.vector_db_name}")     
            st.divider()


# ----------------------
# Existing Databases
# ----------------------
with tabs[1]:
    st.title("🗂 Existing Vector Databases")
    db_dir = "vector_dbs"
    os.makedirs(db_dir, exist_ok=True)

    # List directories inside db_dir (each is a vector DB)
    db_folders = [f for f in Path(db_dir).iterdir() if f.is_dir()]
    if not db_folders:
        st.info("No vector databases found.")
    else:
        for db_folder in db_folders:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"- **{db_folder.stem}** ({db_folder.name})")
            with col2:
                if st.button("🗑 Delete", key=f"delete_{db_folder.name}"):
                    try:
                        shutil.rmtree(db_folder)
                        st.success(f"Deleted: {db_folder.name}")
                    except Exception as e:
                        st.error(f"Error deleting {db_folder.name}: {e}")

