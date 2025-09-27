## AlgorithmX Assessment - RAG App

This project demonstrates a Retrieval-Augmented Generation (RAG) system that integrates Streamlit, FastAPI, SQLAlchemy, PostgreSQL, and Gemini for intelligent document summarization and logging.

---

## 🚀 Project Overview

* **Streamlit:** Provides an interactive frontend for users to upload documents and ask questions.
* **FastAPI:** Handles backend APIs for document processing and logging.
* **SQLAlchemy & PostgreSQL:** Manage and store logs of user interactions.
* **Gemini:** Powers the question-answering mechanism using context from uploaded documents.

---

## 🛠️ Technologies Used

* Python 3.9+
* Streamlit
* FastAPI
* SQLAlchemy
* PostgreSQL
* Gemini (or any LLM for Q&A)
* dotenv (for environment variable management)

---

## Project Structure

```
AlgorithmX-Assessment/
├── app.py                  # Streamlit application
├── api.py                  # FastAPI application
├── utils.py                # RAG utility functions (PDF loading, chunking, Qdrant, etc.)
├── logs.py                 # Logging utilities (JSON + PostgreSQL)
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (MODEL_NAME, GEMINI_MODEL, POSTGRES URL, etc.)
├── Dockerfile_streamlit    # Dockerfile for Streamlit
|── Dockerfile_fastapi      # Dockerfile for FastAPI
├── docker-compose.yaml     # Docker Compose configuration
├── README.md               # This file
├── qdrant_data/            # Local storage for Qdrant
└── logs/                   # Logs folder (auto-generated JSON)
```

---

## Environment Variables

Create a `.env` file in the root directory with the following variables:
```
# Gemini
GEMINI_API_KEY=your_gemini_api_key

# Frontend
FRONTEND_PORT=8501 (For Streamlit)

# Backend
MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
BACKEND_PORT=8000 (For FastAPI)

# PostgreSQL
POSTGRES_USER=your_postgres_username
POSTGRES_PASSWORD=your_postgres_password
POSTGRES_DB=your_postgres_db_name
POSTGRES_PORT=5432 (Default Port)
DATABASE_URL="postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@localhost:${POSTGRES_PORT}/${POSTGRES_DB}"
# Change the @localhost to @postgres for non-local environments

# Qdrant
QDRANT_PORT=6333 (Default Port)
```

Make sure the `.env` file is mentioned in `gitignore` and `dockerignore`.

---

## 📦 Installation

1. Clone the Repo:
```
git clone https://github.com/rajan-bhateja/AlgorithmX-Assessment.git
cd AlgorithmX-Assessment
```

2. Create and activate a virtual environment:
```
python -m venv venv
source venv/bin/activate    # On Windows, use 'venv\Scripts\activate'
```

3. Install dependencies:
```
pip install -r requirements.txt
```

4. Set up environment variables:

Create a `.env` file in the root directory with the variables mentioned in `Environment Variables`

---

## 🧪 Running the Application

1. Start the PostgreSQL Database
```
docker run --name algorithmx-db -e POSTGRES_USER=username -e POSTGRES_PASSWORD=password -e POSTGRES_DB=algorithmx_db -p 5432:5432 -d postgres
```

2. Initialize the Database
```
python -c "from logs import init_db; init_db()"
```

3. Launch the Streamlit App
```
streamlit run app.py
```

4. Launch the FastAPI Backend
```
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

---

## 🧠 How It Works

1. **User Interaction:** Users upload a PDF document and ask a question via the Streamlit interface.
2. **Document Processing:** The document is processed, and the relevant context is retrieved.
3. **Question Answering:** The context and user query are sent to Gemini (or your chosen LLM) for generating an answer.
4. **Logging:** The interaction is logged in both a local JSON file and the PostgreSQL database.

---

## 🧩 Future Enhancements

* Implement pagination and filtering for logs.
* Enhance document processing capabilities (struggling with multiple document processing).
* Integrate additional AI models for improved question answering.
