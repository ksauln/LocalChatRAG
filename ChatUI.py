import os
import time
import streamlit as st
from pathlib import Path
import shutil

st.set_page_config(page_title="LocalChatRAG", layout="wide")

# =============================================================================
# Sidebar config
# =============================================================================
st.sidebar.title("Model & Reasoning Configuration")

# LLM model selector
llm_options = ["llama3.2", "qwen3:4b", "gemma3:4b", "deepseek-r1:7b", "mistral"]  # Customize based on your Ollama setup
llm_model = st.sidebar.selectbox("Select LLM Model", llm_options)

# Embedding model selector
embedding_options = ["nomic-embed-text", "snowflake-arctic-embed"]  # Adjust accordingly
embed_model = st.sidebar.selectbox("Select Embedding Model", embedding_options)

# Reasoning style toggle
reasoning_mode = st.sidebar.radio("Reasoning Method", options=["Chain-of-Thought", "Standard"])
use_cot = reasoning_mode == "Chain-of-Thought"


# =============================================================================
# Main config
# =============================================================================

# Import document processing functions from ChatUpload.py
from ChatUpload import load_uploaded_documents, chunk_text, create_or_update_vector_db
# Import the advanced chain-of-thought LLM query function from ChatLLM.py
from ChatLLM import query_llm_cot, query_llm_norm
# Updated import for SQLChatMessageHistory with new parameters.
from langchain_community.chat_message_histories import SQLChatMessageHistory

# Tabs for chat and DB explorer
tabs = st.tabs(["Chat Interface", "Explore VectorDBs"])

with tabs[0]:
    # =============================================================================
    # Step 1: Document Upload & Processing
    # =============================================================================
    st.title("Document Q&A with LLM and Chat History")
    st.header("Step 1: Upload and Process Documents")


    db_dir = "vector_dbs"
    os.makedirs(db_dir, exist_ok=True)
    existing_dbs = [f.stem for f in Path(db_dir).glob("*.db")]

    use_existing = st.checkbox("Use existing VectorDB", value=True)

    if use_existing and existing_dbs:
        vector_db_name = st.selectbox("Select an existing VectorDB", existing_dbs)
        folder_path = st.text_input("(Optional Local) Enter folder path to update this VectorDB:")
        uploaded_files = st.file_uploader("(Optional Remote) Upload new documents to update this VectorDB", type=['pdf', 'docx', 'txt'], accept_multiple_files=True)
        update_mode = st.checkbox("Update existing VectorDB with new documents")
    else:
        vector_db_name = st.text_input("Enter a name for this VectorDB", value="default_db")
        folder_path = st.text_input("(Local) Enter folder path for this VectorDB:")
        uploaded_files = st.file_uploader("(Remote) Upload your documents (PDF, DOCX, TXT)", type=['pdf', 'docx', 'txt'], accept_multiple_files=True)

    if st.button("Process Documents"):
        documents = []
        db_path = os.path.join(db_dir, f"{vector_db_name}.db")

        if folder_path and os.path.isdir(folder_path):
            with st.spinner("Loading documents from folder..."):
                documents = load_documents(folder_path)

        if not documents and uploaded_files:
            with st.spinner("Processing uploaded files..."):
                documents = load_uploaded_documents(uploaded_files)

        if not documents:
            if use_existing:
                st.info("Using existing VectorDB without updates.")
            else:
                st.error("No valid documents found. Please check your folder path or upload supported files.")
        else:
            st.success(f"Loaded {len(documents)} documents.")
            with st.spinner("Chunking documents..."):
                chunks = chunk_text(documents)
            st.success(f"Created {len(chunks)} text chunks.")

            model_selected_embedding = embed_model
            with st.spinner("Creating/updating vector database..."):
                vectordb = create_or_update_vector_db(chunks, model_selected_embedding, db_path)
            st.session_state.vectordb = vectordb
            st.success("Documents processed and vector DB is ready!")


    # =============================================================================
    # Step 2: Chat History Setup (using SQLChatMessageHistory tied to a user ID)
    # =============================================================================
    st.header("Step 2: Chat History Setup")
    user_id = st.text_input("Enter your User ID for chat history:")

    if user_id:
        # Instantiate or load the chat history for this user.
        # Using the new parameter names: `connection_string` and `session_id`.
        history = SQLChatMessageHistory(
            connection_string="sqlite:///chat_history.db",  # SQLite connection string.
            table_name="chat_history",
            session_id=user_id
        )
        st.session_state.chat_history = history
        st.success(f"Chat history loaded for user: {user_id}")

        # Provide an option to start a new chat (clear the existing chat history)
        if st.button("Start New Chat (Clear History)"):
            try:
                # Assuming SQLChatMessageHistory has a clear() method.
                st.session_state.chat_history.clear()
                st.success("Chat history cleared! You can now start a new chat.")
            except Exception as e:
                st.error(f"Failed to clear chat history: {e}")

    # =============================================================================
    # Step 3: Ask a Question (LLM Query with Chain-of-Thought)
    # =============================================================================
    st.header("Step 3: Ask a Question About the Documents")
    query = st.text_input("Enter your question:")

    if st.button("Get Answer"):
        if "vectordb" not in st.session_state:
            st.error("Please process documents first!")
        elif "chat_history" not in st.session_state:
            st.error("Please set up your chat history by providing your User ID!")
        elif not query:
            st.error("Please enter a question.")
        else:
            # Retrieve the chat history from session_state.
            chat_history = st.session_state.chat_history
            
            # Add the user's query to the chat history.
            chat_history.add_user_message(query)
            
            with st.spinner("Generating answer..."):
                # Call the chain-of-thought LLM query function.
                start_time = time.time()
                if use_cot:
                    result = query_llm_cot(query, st.session_state.vectordb, model_selected_llm=llm_model)
                else:
                    result = query_llm_norm(query, st.session_state.vectordb, model_selected_llm=llm_model)
                elapsed_time = time.time() - start_time
                minutes = int(elapsed_time // 60)
                seconds = int(elapsed_time % 60)


            # Save the LLM's response to the chat history.
            chat_history.add_ai_message(result["answer"])
            
            # Display the LLM's answer and its sources.
            st.subheader("Answer")
            st.write(result["answer"])
            
            st.subheader("Sources")
            for source in result["sources"]:
                st.write(f"- {source}")
            
            
            st.caption(f"⏱️ Response time: {minutes} min {seconds} sec")     
            # =============================================================================
            # Display the complete chat history for the current user.
            # =============================================================================
            st.header("Chat History")
            # Assuming chat_history.messages returns a list of message objects with attributes `type` and `content`.
            for message in chat_history.messages:
                if message.type == "human":
                    st.markdown(f"**User:** {message.content}")
                elif message.type == "ai":
                    st.markdown(f"**LLM:** {message.content}")

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
