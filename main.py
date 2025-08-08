from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, HttpUrl
from typing import List, Optional, Dict, Any
import logging
import os
import sys
from langchain.schema import Document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import your existing modules
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
import requests
from io import BytesIO

# Initialize FastAPI app
app = FastAPI(
    title="Bajaj Policy Assistant API",
    description="API for processing insurance policy documents and answering questions",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()
API_TOKEN = "9febf0ce1fb01389f2b3de3e61ef4afad05bbffd463725a337323121898d3b8a"  # Move to environment variables

# Models
class QueryRequest(BaseModel):
    documents: str  # URL to the document
    questions: List[str]

class AnswerResponse(BaseModel):
    answers: List[str]

def create_empty_faiss_index(embeddings):
    """Create an empty FAISS index with a single dummy document"""
    return FAISS.from_documents(
        [Document(page_content="Initial document")], 
        embeddings
    )

# Initialize global components
embeddings = HuggingFaceEmbeddings()
vectorstore = None

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify the provided bearer token"""
    if credentials.credentials != API_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

def load_document_from_url(url: str):
    """Load document from URL"""
    try:
        # Download the file
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Determine file type from content type or URL
        content_type = response.headers.get('content-type', '')
        file_extension = url.split('.')[-1].lower()
        
        # Load based on file type
        if 'pdf' in content_type or file_extension == 'pdf':
            loader = PyPDFLoader(BytesIO(response.content))
        elif 'word' in content_type or file_extension in ['docx', 'doc']:
            loader = UnstructuredWordDocumentLoader(BytesIO(response.content))
        elif 'text' in content_type or file_extension in ['txt', 'text']:
            loader = TextLoader(BytesIO(response.content))
        else:
            raise ValueError(f"Unsupported document type: {content_type}")
            
        return loader.load()
    except Exception as e:
        logger.error(f"Error loading document: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error loading document: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    global vectorstore
    try:
        # Initialize FAISS index if it exists, otherwise create an empty one
        if os.path.exists("vectorstore"):
            try:
                # Load with dangerous deserialization allowed since we trust our own files
                vectorstore = FAISS.load_local(
                    "vectorstore", 
                    embeddings, 
                    allow_dangerous_deserialization=True
                )
                logger.info("Loaded existing FAISS index")
            except Exception as e:
                logger.warning(f"Error loading FAISS index, creating new one: {str(e)}")
                vectorstore = create_empty_faiss_index(embeddings)
        else:
            vectorstore = create_empty_faiss_index(embeddings)
            logger.info("Created new FAISS index")
    except Exception as e:
        logger.error(f"Error initializing vectorstore: {str(e)}")
        raise

@app.post("/hackrx/run", response_model=AnswerResponse)
def process_queries(
    request: QueryRequest,
    token: str = Depends(verify_token)
):
    """
    Process document and answer questions
    
    Args:
        request: QueryRequest containing document URL and list of questions
        
    Returns:
        AnswerResponse with list of answers
    """
    try:
        # Load and process the document
        documents = load_document_from_url(request.documents)
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)
        
        # Update FAISS index with new document chunks
        global vectorstore
        if len(chunks) > 0:
            new_vectorstore = FAISS.from_documents(chunks, embeddings)
            if vectorstore is not None:
                vectorstore.merge_from(new_vectorstore)
            else:
                vectorstore = new_vectorstore
            
            # Save the updated index with safe serialization
            try:
                vectorstore.save_local(
                    "vectorstore",
                    safe_serialization=True
                )
            except Exception as e:
                logger.error(f"Error saving FAISS index: {str(e)}")
                raise
        
        # Process each question
        answers = []
        for question in request.questions:
            try:
                # Perform similarity search
                docs = vectorstore.similarity_search(question, k=3)
                
                # For now, just return the content of the most relevant document
                if docs:
                    answers.append(docs[0].page_content[:500])  # Limit response length
                else:
                    answers.append("No relevant information found.")
            except Exception as e:
                logger.error(f"Error processing question '{question}': {str(e)}")
                answers.append(f"Error processing question: {str(e)}")
        
        return AnswerResponse(answers=answers)
        
    except Exception as e:
        logger.error(f"Error in process_queries: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
