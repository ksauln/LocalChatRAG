import os
import glob
import time
import tempfile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_core.runnables import RunnablePassthrough
from langchain.schema import format_document, Document



def load_uploaded_documents(uploaded_files):
    all_docs = []
    for uploaded_file in uploaded_files:
        suffix = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
        
        if suffix == ".pdf":
            loader = PyPDFLoader(tmp_path)
        elif suffix == ".docx":
            loader = Docx2txtLoader(tmp_path)
        elif suffix == ".txt":
            loader = TextLoader(tmp_path)
        else:
            continue  # Skip unsupported files

        docs = loader.load()
        # Attach original filename as metadata
        for doc in docs:
            doc.metadata["source"] = uploaded_file.name
        all_docs.extend(docs)
        os.remove(tmp_path)  # Clean up temp file

    return all_docs



# not currently used
'''
def load_documents(root_folder_path):
    """Recursively load and categorize Word, PDF, and TXT documents from a folder and its subdirectories,
    storing filenames in metadata.
    """
    print('Scanning for documents in all subdirectories...')
    documents = []
    
    # Recursively search for DOCX, PDF, and TXT files
    docx_files = glob.glob(os.path.join(root_folder_path, '**', '*.docx'), recursive=True)
    pdf_files = glob.glob(os.path.join(root_folder_path, '**', '*.pdf'), recursive=True)
    txt_files = glob.glob(os.path.join(root_folder_path, '**', '*.txt'), recursive=True)
    
    if not docx_files and not pdf_files and not txt_files:
        print("No DOCX, PDF, or TXT files found in the folder or its subdirectories.")
        return documents

    # Load DOCX files
    for file_path in docx_files:
        print(f"Loading Word document: {file_path}")
        try:
            loader = Docx2txtLoader(file_path)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = os.path.basename(file_path)
            documents.extend(docs)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    # Load PDF files
    for file_path in pdf_files:
        print(f"Loading PDF document: {file_path}")
        try:
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = os.path.basename(file_path)
            documents.extend(docs)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    # Load TXT files
    for file_path in txt_files:
        print(f"Loading text document: {file_path}")
        try:
            loader = TextLoader(file_path)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = os.path.basename(file_path)
            documents.extend(docs)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    print(f'Documents Loaded! Total: {len(documents)}')
    return documents
'''

def chunk_text(documents, chunk_size=300, chunk_overlap=125):
    """Split the text from documents into smaller chunks to facilitate processing.
    
    - Small Chunks (e.g., 256-512 tokens):
        • Suitable for concise, factual retrieval.
        • Reduces memory usage but may miss context.
   - Medium Chunks (e.g., 512-1024 tokens):
        • A balance between context and processing efficiency.
   - Large Chunks (e.g., 1024+ tokens):
        • Preserve more context but can introduce irrelevant information.
    
    Typical overlap ranges: 10-30% of the chunk size.
        • For highly interrelated content (e.g., legal, medical), larger overlaps (30-50%) might be needed.

    Can expierment with different chunking techniques to see how it changes results. 
"""
    print('Chunking text...')
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(documents)
    print('Chunking Complete!')
    return chunks

def check_vector_db_exists(db_path):
    """Check if the vector database exists."""
    return os.path.exists(db_path) and os.path.isdir(db_path)

def create_or_update_vector_db(chunks, model_selected_embedding, db_path):
    """Create or update the vector database while ensuring metadata (filename) is preserved."""
    start_time = time.time()
    print('Checking for existing vectorDB...')
    embeddings_model = OllamaEmbeddings(model=model_selected_embedding)

    if check_vector_db_exists(db_path):
        print("VectorDB exists. Loading and updating with new data...")
        vectordb = Chroma(persist_directory=db_path, embedding_function=embeddings_model)
        
        # Extract texts and metadata (filenames) for new documents
        texts = [doc.page_content for doc in chunks if doc.page_content.strip()]
        metadatas = [{"source": doc.metadata.get("source", f"document_{i}")} for i, doc in enumerate(chunks)]
        
        vectordb.add_texts(texts=texts, metadatas=metadatas)
        print("VectorDB updated with new documents!")
    else:
        print("No existing VectorDB found. Creating a new one...")
        texts = [doc.page_content for doc in chunks if doc.page_content.strip()]
        metadatas = [{"source": doc.metadata.get("source", f"document_{i}")} for i, doc in enumerate(chunks)]
        
        vectordb = Chroma.from_texts(
            texts=texts,
            embedding=embeddings_model,
            metadatas=metadatas,
            persist_directory=db_path
        )

        print(f"New VectorDB created! Time taken: {time.time() - start_time:.2f} seconds")

    return vectordb
