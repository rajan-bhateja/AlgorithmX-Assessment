from datetime import datetime

import streamlit as st
from utils import *
from logs import log_response


# Streamlit page config
st.set_page_config(page_title="AlgorithmX RAG App", layout="wide")
st.title("AlgorithmX RAG App")
st.caption("Powered by Gemini and Qdrant")

# Initialize session state for logs
if "logs" not in st.session_state:
    st.session_state["logs"] = []

# User info
user_name = st.text_input("Enter your name:")

home_tab, conversation_tab = st.tabs(["Home", "Conversation"])

# PDF upload + query
with home_tab:
    with st.form("upload_form"):
        uploaded_files = st.file_uploader(
            "Upload PDF(s)", type="pdf", accept_multiple_files=True
        )
        user_prompt = st.text_input(
            "Ask questions about your PDF(s)",
            placeholder="Explain the Transformer architecture"
        )
        submit = st.form_submit_button("Upload & Query")

    global answer

    if submit:
        if not user_name:
            st.error("Please enter your name.")
        elif not uploaded_files:
            st.error("Please upload at least one PDF.")
        else:
            for file in uploaded_files:
                if file.type.endswith("pdf"):
                    # Load PDF
                    docs = load_pdf_file(file)
                    # Chunk docs
                    chunks = chunk_docs(docs)
                    # Store in Qdrant
                    vectorstore = store_in_qdrant(chunks)
                    # Retrieve relevant chunks
                    top_chunks = retrieve_similar_chunks(vectorstore, user_prompt)
                    # Generate LLM answer
                    answer = generate_answer_from_chunks(top_chunks, user_prompt)
                    if answer:
                        st.success(f"Answer generated for {file.name}!\n\nCheck the Conversations Tab for more!")
                        with conversation_tab:
                            # Display answer
                            st.subheader(f"Gemini Response:")
                            st.write(answer)

                        # Create log entry
                        now_str = datetime.now().strftime("%d%m%Y_%H%M%S")
                        current_login = f"{now_str}_{user_name}"
                        log_entry = {
                            "source": "streamlit",
                            "current_login": current_login,
                            "user_prompt": user_prompt,
                            "user_uploads": file.name,
                            "answer": answer
                        }
                        log_path = log_response(
                            source="streamlit",
                            current_login=current_login,
                            user_name=user_name,
                            user_prompt=user_prompt,
                            user_uploads=file.name,
                            answer=answer
                        )
                        st.info(f"Logs saved to {log_path} and stored in PostgreSQL")

                        # Add to session state
                        st.session_state.logs.append(log_entry)
