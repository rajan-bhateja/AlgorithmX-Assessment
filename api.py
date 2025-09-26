from fastapi import FastAPI, UploadFile, File, Form
from tempfile import NamedTemporaryFile
from datetime import datetime
from logs import log_response

from utils import *
from langchain_community.vectorstores import Qdrant

app = FastAPI()

# Keep vectorstore reference globally
vectorstore: Qdrant | None = None


@app.post("/upload_pdf/")
async def upload_pdf_endpoint(
    file: UploadFile = File(...),
    user_prompt: str = Form(...),
    user_name: str = Form(...)
):
    global vectorstore

    # Current Time
    now_str = datetime.now().strftime("%d%m%Y_%H%M%S")
    current_login = f"{now_str}_{user_name}"

    # Save FastAPI file to a real temporary file
    with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    # Use your existing loader with the temp file path
    docs = load_pdf_file(open(tmp_path, "rb"))
    chunks = chunk_docs(docs)

    # Store in Qdrant
    vectorstore = store_in_qdrant(
        chunks=chunks,
        # Disabled these due to Internal Server Error 500
        # embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        # collection_name="word_embeddings"
    )

    # Retrieve relevant chunks
    top_chunks = retrieve_similar_chunks(vectorstore, user_prompt)

    # Generate LLM answer
    answer = generate_answer_from_chunks(top_chunks, user_prompt)

    # Logs Handling
    user_profile_info = {
        "source": "fastapi",
        "current_login": current_login,
        "user_prompt": user_prompt,
        "user_uploads": file.filename,
        "answer": answer
    }

    log_path = log_response(
        source="fastapi",
        current_login=current_login,
        user_name=user_name,
        user_prompt=user_prompt,
        user_uploads=file.filename,
        answer=answer,
    )

    return {
        "status": "success",
        "filename": file.filename,
        "user_prompt": user_prompt,
        "answer": answer,
    }
