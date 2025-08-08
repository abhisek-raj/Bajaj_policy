import os
import tempfile
from pathlib import Path
from typing import List, Optional
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

def load_documents_from_file(file_path: str) -> List[Document]:
    """Load documents from a file, supporting both PDF and TXT formats."""
    file_path = Path(file_path)
    try:
        if file_path.suffix.lower() == '.pdf':
            loader = PyPDFLoader(str(file_path))
        else:  # Assuming .txt or other text files
            loader = TextLoader(str(file_path))
        
        documents = loader.load()
        
        # Add filename as title in metadata for each document
        for doc in documents:
            if not hasattr(doc, 'metadata'):
                doc.metadata = {}
            doc.metadata['title'] = file_path.stem  # Use filename without extension as title
            
        return documents
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return []

def create_chunks(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 100) -> List[Document]:
    """Split documents into chunks for processing."""
    if not documents:
        return []
        
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return text_splitter.split_documents(documents)

def create_vector_store(documents: List[Document], vectorstore_path: str = "vectorstore/db_faiss") -> Optional[FAISS]:
    """Create or update a FAISS vector store from documents."""
    if not documents:
        print("No documents to process for vector store.")
        return None
        
    try:
        # Initialize embeddings model
        embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Create or load existing vector store
        if os.path.exists(vectorstore_path) and os.listdir(vectorstore_path):
            print(f"Loading existing vector store from {vectorstore_path}")
            db = FAISS.load_local(vectorstore_path, embedding_model, allow_dangerous_deserialization=True)
            print(f"Adding {len(documents)} new documents to existing vector store")
            db.add_documents(documents)
        else:
            print(f"Creating new vector store at {vectorstore_path}")
            os.makedirs(vectorstore_path, exist_ok=True)
            db = FAISS.from_documents(documents, embedding_model)
        
        # Save the updated vector store
        db.save_local(vectorstore_path)
        print(f"Vector store saved successfully with {db.index.ntotal} vectors")
        return db
        
    except Exception as e:
        print(f"Error creating/updating vector store: {str(e)}")
        return None

def process_data_directory(data_dir: str = "data") -> List[Document]:
    """Process all PDF and TXT files in the data directory."""
    data_dir = Path(data_dir)
    if not data_dir.exists():
        print(f"Data directory {data_dir} does not exist")
        return []
    
    all_documents = []
    
    # Process all PDF and TXT files in the data directory
    for file_path in data_dir.glob("*.[pP][dD][fF]"):  # PDF files
        print(f"Processing PDF file: {file_path.name}")
        documents = load_documents_from_file(file_path)
        if documents:
            all_documents.extend(documents)
    
    for file_path in data_dir.glob("*.[tT][xX][tT]"):  # Text files
        print(f"Processing text file: {file_path.name}")
        documents = load_documents_from_file(file_path)
        if documents:
            all_documents.extend(documents)
    
    print(f"Processed {len(all_documents)} total documents from {data_dir}")
    return all_documents

def main():
    # Process all documents in the data directory
    documents = process_data_directory("data")
    
    if not documents:
        print("No documents found to process. Exiting.")
        return
    
    # Create chunks from documents
    print("Splitting documents into chunks...")
    chunks = create_chunks(documents)
    print(f"Created {len(chunks)} chunks from {len(documents)} documents")
    
    # Remove existing vector store to ensure clean rebuild
    vectorstore_path = "vectorstore/db_faiss"
    if os.path.exists(vectorstore_path):
        import shutil
        print(f"Removing existing vector store at {vectorstore_path}")
        shutil.rmtree(vectorstore_path)
    
    # Create a new vector store with updated metadata
    create_vector_store(chunks)
    print("Processing complete!")
    print("Vector store has been rebuilt with document titles in metadata.")

if __name__ == "__main__":
    main()