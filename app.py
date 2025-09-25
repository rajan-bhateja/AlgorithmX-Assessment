from datetime import datetime
from dotenv import load_dotenv
import tempfile

import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

from google import genai
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader

from langchain_qdrant import QdrantVectorStore
from langchain_community.vectorstores import Qdrant
from qdrant_client.grpc import VectorParams, Distance
from qdrant_client import QdrantClient


# Load the environment variables (API Keys)
load_dotenv()

# VARIABLES (Change accordingly)
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
gemini_model = "gemini-2.5-flash"
collection_name = "word_embeddings"


def load_pdf_file(uploaded_file: UploadedFile) -> list[Document]:
    """
    Load PDF into temporary files to be parsed by PyMuPDFLoader.
    Returns a Langchain Document object.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name
    print(f"Temporary file created at {temp_file_path}")

    # Load the Temp file into PyMuPDFLoader
    # Single mode to parse the entire PDF as a single document
    doc = PyMuPDFLoader(temp_file_path, mode="single").load()
    return doc


def chunk_docs(documents, chunk_size: int = 500, chunk_overlap: int = 100) -> list[Document]:
    """Split Documents into specified chunk sizes"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len)
    raw_chunks = text_splitter.split_documents(documents)
    return raw_chunks


def store_in_qdrant(chunks: list[Document], embedding_model: str, collection_name: str) -> Qdrant:
    """Encode and store chunks in Qdrant"""

    # Embeddings
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    # Local persistence
    client = QdrantClient(url="http://localhost:6333")

    # Check for any existing collections
    existing_collections = [col.name for col in client.get_collections().collections]
    if collection_name not in existing_collections:
        client.create_collection(
            collection_name=collection_name,
            force_recreate=True
            # vectors_config=VectorParams(size=384)
        )

    vectorstore = Qdrant.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        force_recreate=True
    )
    return vectorstore


def retrieve_similar_chunks(
        vectorstore: Qdrant,
        query: str,
        embedding_model: str = embedding_model_name,
        top_k: int = 3
) -> list[Document]:
    """
        Embed a user query and retrieve top-k most similar document chunks from Qdrant.

        Args:
            vectorstore (Qdrant): The Qdrant vector store containing PDF chunks.
            query (str): User query to search for.
            embedding_model_name (str): HuggingFace wrapped Embedding Models. Default: sentence-transformers/all-MiniLM-L6-v2.
            top_k (int): Number of top similar chunks to retrieve.

        Returns:
            List[Document]: Top-k most similar Document chunks.
    """
    # Initialize the embedding model
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    # Retrieve top-k results
    results = vectorstore.similarity_search(query, top_k)

    return results


def generate_answer_from_chunks(
        chunks: list,
        user_query: str,
        llm_model: str = gemini_model
) -> str:
    """Generate answer using LLM from retrieved chunks"""
    context = "\n\n".join([chunk.page_content for chunk in chunks])
    prompt = f"Answer the following question based only on the context:\n\n{context}\n\nQuestion: {user_query}\n\nAnswer:"

    # Using the Google's own Gemini API
    llm = genai.Client()
    response = llm.models.generate_content(
        model=llm_model,
        contents=prompt,
    )

    return response.candidates[0].content.parts[0].text


# STREAMLIT UI
st.set_page_config(page_title="AlgorithmX Assessment - RAG App", layout="wide")
st.title("AlgorithmX Assessment - RAG App")
st.caption("Powered by Gemini and Qdrant")

user_profile_info = {}

with st.popover("Your name"):
    user_name = st.text_input("Enter your name:")
    now = datetime.now()
    now_str = now.strftime("%d%m%Y %H:%M:%S")
    current_login = now_str + " " + user_name
    user_profile_info["current_login"] = current_login

with st.form(key="my_form"):
    user_uploads = st.file_uploader(
        "Upload your PDF(s)", type="pdf", accept_multiple_files=True
    )
    user_profile_info["user_uploads"] = user_uploads
    user_prompt = st.text_input(
        "Ask questions about your PDF(s)",
        placeholder="Explain the Transformer architecture"
    )
    user_profile_info["user_prompt"] = user_prompt

    submit = st.form_submit_button("Process PDFs & Query")

    if submit:
        if not user_uploads:
            st.error("Please upload at least one PDF to continue!")
        else:
            # Load and chunk all PDFs
            all_docs = []
            for file in user_uploads:
                if file.type.endswith("pdf"):
                    docs = load_pdf_file(file)
                    all_docs.extend(docs)
            chunks = chunk_docs(all_docs)
            st.success(f"Created {len(chunks)} chunks from uploaded PDFs.")

            # Store chunks in Qdrant
            vectorstore = store_in_qdrant(chunks, embedding_model_name, collection_name)
            st.success(f"Chunks stored in Qdrant collection '{collection_name}'.")

            # Retrieve and generate answers
            if user_prompt:
                with st.spinner("Retrieving relevant chunks..."):
                    top_chunks = retrieve_similar_chunks(vectorstore, user_prompt)
                st.subheader("Top Retrieved Chunks")
                for idx, chunk in enumerate(top_chunks, 1):
                    st.markdown(f"**Chunk {idx}:** {chunk.page_content[:500]}...")

                with st.spinner("Generating answer using Gemini..."):
                    answer = generate_answer_from_chunks(top_chunks, user_prompt)
                st.subheader("LLM Answer")
                st.write(answer)
