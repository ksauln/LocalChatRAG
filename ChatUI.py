import os
import time
import streamlit as st

# Import document processing functions from ChatUpload.py
from ChatUpload import load_documents, chunk_text, create_or_update_vector_db

# Import the advanced chain-of-thought LLM query function from ChatLLM.py
from ChatLLM import query_llm_cot

# Updated import for SQLChatMessageHistory with new parameters.
from langchain_community.chat_message_histories import SQLChatMessageHistory



#models selected
embed_model = "nomic-embed-text" # Update with your embedding model.
llm_model = "llama3.2"  #Update with your actual LLM model name.

#model options on my computer: llama3.2, phi4, deepseek-r1

# =============================================================================
# Step 1: Document Upload & Processing
# =============================================================================
st.title("Document Q&A with LLM and Chat History")
st.header("Step 1: Upload and Process Documents")

folder_path = st.text_input("Enter the folder path containing your DOCX/PDF documents:")

if st.button("Process Documents"):
    if not folder_path or not os.path.exists(folder_path):
        st.error("Please enter a valid folder path.")
    else:
        with st.spinner("Loading documents..."):
            documents = load_documents(folder_path)
        if not documents:
            st.warning("No documents found in the folder.")
        else:
            st.success(f"Loaded {len(documents)} documents.")
            with st.spinner("Chunking documents..."):
                chunks = chunk_text(documents)
            st.success(f"Created {len(chunks)} text chunks.")
            
            # Create or update the vector database.
            db_path = "vector_db"  # Change this directory if needed.
            model_selected_embedding = embed_model
            with st.spinner("Creating/updating vector database..."):
                vectordb = create_or_update_vector_db(chunks, model_selected_embedding, db_path)
            st.success("Vector Database is ready!")
            
            # Save the vector DB in Streamlit's session_state for later use.
            st.session_state.vectordb = vectordb

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
            result = query_llm_cot(query, st.session_state.vectordb, model_selected_llm=llm_model)
        
        # Save the LLM's response to the chat history.
        chat_history.add_ai_message(result["answer"])
        
        # Display the LLM's answer and its sources.
        st.subheader("Answer")
        st.write(result["answer"])
        
        st.subheader("Sources")
        for source in result["sources"]:
            st.write(f"- {source}")
        
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