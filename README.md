# Bajaj Policy Assistant API

A FastAPI service for processing insurance policy documents and answering questions using LLM-powered semantic search.

## Features

- Process PDF, DOCX, and text documents from URLs
- Answer natural language questions about policy documents
- Uses FAISS for efficient semantic search
- Secure API with Bearer token authentication
- Containerized with Docker for easy deployment

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements-api.txt
   ```

2. **Environment Variables**:
   Create a `.env` file with your API token:
   ```
   API_TOKEN=your_api_token_here
   HUGGINGFACEHUB_API_TOKEN=your_hf_token_here
   ```

## Running the API

### Development
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Production with Docker
```bash
docker build -t bajaj-policy-api .
docker run -p 8000:8000 -e API_TOKEN=your_token_here bajaj-policy-api
```

## API Documentation

Once running, access the interactive API docs at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### POST /hackrx/run

Process a document and answer questions about it.

**Request Body**:
```json
{
  "documents": "https://example.com/policy.pdf",
  "questions": [
    "What is the grace period for premium payment?",
    "What is covered under this policy?"
  ]
}
```

**Response**:
```json
{
  "answers": [
    "The grace period is 30 days.",
    "This policy covers medical expenses, hospitalization, and surgery."
  ]
}
```

## Development

### Running Tests
```bash
pytest
```

### Code Formatting
```bash
black .
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
