import tempfile
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.grpc import VectorParams, Distance
from google import genai
from dotenv import load_dotenv


# Load Environment Variables
load_dotenv()


# CONFIG
# embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
gemini_model = "gemini-2.5-flash"
collection_name = "word_embeddings"
qdrant_path = "./qdrant_data"


def load_pdf_file(uploaded_file) -> list[Document]:
    """Save uploaded file temporarily and load as Documents"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.file.read() if hasattr(uploaded_file, "file") else uploaded_file.read())
        temp_file_path = temp_file.name
    return PyMuPDFLoader(temp_file_path, mode="single").load()


def chunk_docs(documents, chunk_size: int = 500, chunk_overlap: int = 200) -> list[Document]:
    """Split Documents into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len)
    return text_splitter.split_documents(documents)


def store_in_qdrant(chunks: list[Document], path: str = qdrant_path) -> Qdrant:
    """Encode and store chunks in Qdrant"""
    embeddings = HuggingFaceEmbeddings()
    client = QdrantClient(path=path)

    existing_collections = [col.name for col in client.get_collections().collections]
    if collection_name not in existing_collections:
        client.create_collection(
            collection_name=collection_name,
            force_recreate=True
            # vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )

    vectorstore = Qdrant.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        force_recreate=True
    )
    return vectorstore


def retrieve_similar_chunks(vectorstore: Qdrant, query: str, top_k: int = 5) -> list[Document]:
    """Retrieve top-k most similar chunks"""
    return vectorstore.similarity_search(query, top_k)


def generate_answer_from_chunks(chunks: list[Document], user_query: str, llm_model: str = gemini_model) -> str:
    """Generate answer using Gemini"""
    context = "\n\n".join([chunk.page_content for chunk in chunks])
    prompt = (
        f"Answer the following question based only on the context:\n\n"
        f"{context}\n\n"
        f"Question: {user_query}\n"
        f"Answer:"
    )

    # Using Google's own Gemini API
    llm = genai.Client()
    response = llm.models.generate_content(
        model=llm_model,
        contents=prompt,
    )
    return response.candidates[0].content.parts[0].text