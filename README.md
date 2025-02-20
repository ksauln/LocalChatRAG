# LocalChatRAG

# Chat App with Document Ingestion, LLM Querying, and Persistent Chat History

This project provides a web-based chat application built with Streamlit that allows users to:
- **Ingest Documents:** Recursively scan a folder (and its subdirectories) for DOCX, PDF, and TXT files.
- **Process and Index:** Load and chunk the documents, then store them in a vector database using embeddings.
- **Query with an LLM:** Use a chain-of-thought (CoT) approach to query the documents via an LLM (powered by Ollama).
- **Persist Chat History:** Save conversation history using SQLChatMessageHistory, tied to a user ID, with the option to clear history and start a new chat.

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
  - [Installing Ollama](#installing-ollama)
  - [Installing Python Packages](#installing-python-packages)
- [Project Structure](#project-structure)
- [Detailed Description](#detailed-description)
  - [Document Ingestion and Processing (`ChatUpload.py`)](#document-ingestion-and-processing-chatuploadpy)
  - [LLM Querying with Chain-of-Thought (`ChatLLM.py`)](#llm-querying-with-chain-of-thought-chatllmpy)
  - [Chat Interface and Persistent History (`ChatUI.py`)](#chat-interface-and-persistent-history-apppy)
- [How to Run the App](#how-to-run-the-app)
- [Customization and Troubleshooting](#customization-and-troubleshooting)

## Overview

The chat app allows users to easily upload a folder containing documents, process them into a searchable vector database, and then ask questions using a sophisticated LLM reasoning process. Each conversation is stored in a SQL-backed chat history, making it possible to review past interactions or clear them to start anew.

## Requirements

- **Python 3.8+**
- **Ollama** – A tool to run LLMs locally.
- Python packages:
  - `streamlit`
  - `langchain` and `langchain_community`
  - `chromadb` (or another vector DB backend)
  - Document loader packages like `python-docx` and `PyPDF2`
  - Additional dependencies as required

## Installation

### Installing Ollama

1. Visit the [Ollama website](https://www.ollama.com) and follow the installation instructions for your operating system.
2. Ensure Ollama is running on your machine, as it is required for LLM interactions.

### Installing Python Packages

It is recommended to use a virtual environment. For example:

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install required packages
pip install streamlit langchain langchain_community chromadb python-docx PyPDF2
```

## Project Structure

```
.
├── ChatUpload.py     # Handles document loading, chunking, and vector DB creation/update.
├── ChatLLM.py        # Contains the advanced LLM query function with chain-of-thought reasoning.
├── ChatUI.py         # The Streamlit app that ties everything together.
└── README.md         # This documentation file.
```

## Detailed Description

### Document Ingestion and Processing (`ChatUpload.py`)

**Purpose:**
- To scan a specified root folder and all its subdirectories for DOCX, PDF, and TXT files, load them using dedicated loaders, and store file metadata (e.g., filename).

**Key Functions:**
- `load_documents(root_folder_path)`: Recursively searches the folder for files with extensions .docx, .pdf, and .txt and loads them.
- `chunk_text(documents, chunk_size=300, chunk_overlap=125)`: Splits loaded documents into smaller chunks to optimize for memory usage and efficient retrieval.
- `create_or_update_vector_db(chunks, model_selected_embedding, db_path)`: Creates or updates a vector database with the document chunks using embeddings.

### LLM Querying with Chain-of-Thought (`ChatLLM.py`)

**Purpose:**
- To generate answers to user queries using an LLM with a multi-step chain-of-thought approach:
  1. **Generate Reasoning Steps:** Create a step-by-step plan for answering the question.
  2. **Execute and Verify:** Use the plan along with relevant context from the vector database to generate and refine the answer.

**Key Function:**
- `query_llm_cot(question, db, model_selected_llm)`: Retrieves context from the vector DB, constructs prompts for reasoning, and returns the final answer along with document sources.

### Chat Interface and Persistent History (`ChatUI.py`)

**Purpose:**
- Provides a user-friendly web interface where users can:
  - Specify a folder path for document ingestion.
  - Process the documents and build the vector database.
  - Set up a user-specific chat history using SQLChatMessageHistory (with new parameters like `connection_string` and `session_id`).
  - Ask questions and view answers, complete with conversation history.
  - Clear the chat history to start a new conversation.

**Key Features:**
- **Document Processing:** Input a folder path and process all DOCX, PDF, and TXT files.
- **Persistent Chat History:** Save user conversations in a SQL database.
- **New Chat Option:** Clear existing history to start a fresh conversation.
- **LLM Querying:** Submit a query, retrieve context, and display the LLM-generated answer with source details.

## How to Run the App

1. Open a Terminal or VS Code Integrated Terminal.
2. Navigate to your project directory:
   ```bash
   cd path/to/your/project
   ```
3. Activate the Virtual Environment:
   ```bash
   # On macOS/Linux:
    source venv/bin/activate
    # On Windows:
    venv\Scripts\activate
   ``` 
4. Run the Streamlit App:
   ```bash
   streamlit run ChatUI.py
   ```
   
   **Optional:** To run the app on a specific host/port (e.g., accessible externally):
   ```bash
   streamlit run ChatUI.py --server.address 0.0.0.0 --server.port 8501
   ```
5. Open your Browser:
   - The terminal will display a local URL (e.g., `http://localhost:8501`). Navigate to this URL to use the app.

## Customization and Troubleshooting

- **Model Settings:**
  - Update model names in the code (e.g., `your-embedding-model` and `your-llm-model`) to match your Ollama configuration.

- **Database Configuration:**
  - Modify the `db_path` for the vector database or change the SQLite connection string in the chat history configuration as needed.

- **Clearing Chat History:**
  - The app provides an option to clear chat history. Ensure that your version of `SQLChatMessageHistory` (imported from `langchain_community.chat_message_histories`) supports the `clear()` method. If not, refer to the LangChain documentation for guidance.

- **Troubleshooting:**
  - If you encounter module import warnings or errors (e.g., deprecation messages), verify that you are using the latest versions of the required packages and refer to the updated LangChain documentation.
