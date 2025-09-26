## AlgorithmX Assessment - RAG App

A Retrieval-Augmented Generation (RAG) app that allows users to upload PDFs, query them using a Large Language Model (Gemini), and retrieve answers based on document content.
The app uses Qdrant for vector storage, PostgreSQL for logging, and supports both Streamlit UI and FastAPI endpoints.

---

## Features

* Upload PDF documents and process them into document chunks.
* Store embeddings in Qdrant for semantic search.
* Retrieve relevant chunks and generate answers using the Gemini API.
* Keep automatic logs in a `logs/` folder and PostgreSQL database.
* Support for conversational queries.
* Streamlit UI for interactive use.
* FastAPI endpoints for programmatic access.

---

## Tech Stack

* Python: Core language.
* LangChain: Document loading, chunking, and embeddings.
* Qdrant: Vector database for embeddings.
* Sentence Transformers: HuggingFace embeddings model.
* Gemini API: LLM for answer generation.
* PostgreSQL: Logging and analytics.
* Streamlit: Frontend UI.
* FastAPI: API endpoints.
* Docker & Docker Compose: Containerized services.

---

## Project Structure

```
AlgorithmX-Assessment/
├── app.py                  # Streamlit application
├── api.py                  # FastAPI application
├── utils.py                # RAG utility functions (PDF loading, chunking, Qdrant)
├── logs.py                 # Logging utilities (JSON + PostgreSQL)
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (MODEL_NAME, GEMINI_MODEL)
├── Dockerfile              # Dockerfile for FastAPI and Streamlit
├── docker-compose.yaml     # Docker Compose configuration
├── qdrant_data/            # Local storage for Qdrant
└── logs/                   # Logs folder (auto-generated JSON)
```

---

## Environment Variables

Create a `.env` file in the root directory:
```
MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
GEMINI_API_KEY=your_gemini_api_key
POSTGRES_USER=admin
POSTGRES_PASSWORD=admin123
POSTGRES_DB=algorithmx_logs
```

---

## Setup Instructions

1. Clone the Repo:
```
git clone https://github.com/rajan-bhateja/AlgorithmX-Assessment.git
cd AlgorithmX-Assessment
```
2. Install Dependencies:
```
pip install -r requirements.txt
```
3. Run the Dockerfile
```
docker-compose up --build
```
* PostgreSQL: `localhost:5432`
* Qdrant: `localhost:6333`
* FastAPI: `localhost:8000/docs`
* Streamlit: `localhost:8501`

---

## Usage

### Streamlit

1. Enter your name.
2. Upload one or multiple PDFs.
3. Ask a question about the uploaded PDFs.
4. View answers from Gemini and automatically generate logs.

### FastAPI

1. Endpoint: `upload_pdf/`
2. Method: POST
3. Form parameters:
   * `file` -> PDF file
   * `user_name` -> Your name
   * `user_prompt` -> Question about the PDF
4. Returns JSON with `answer`

---

## Logging

* Logs are automatically saved in the `logs/` folder as JSON.
* Logs are also stored in PostgreSQL with fields:
1. `source`
2. `current_login`
3. `user_name`
4. `user_prompt`
5. `user_uploads`
6. `answer`
7. `created_at`

---
