import streamlit as st
import tempfile
import os
from rag_engine import (
    EmbeddingClient,
    build_rag_from_pdf,
    query_rag,
    generate_answer_with_hf
)

st.set_page_config(page_title="PDF Insight AI", layout="wide")

st.title("📄 PDF Insight AI — Intelligent PDF Question Answering")

st.write("""
Upload any PDF and ask questions.

This app uses:
- Local **MiniLM embeddings** (fast, accurate)
- **Mistral-7B-Instruct** for answer generation via Hugging Face
""")

# Sidebar
st.sidebar.header("⚙ Settings")
hf_key = st.sidebar.text_input("Hugging Face API Key", type="password")

if not hf_key:
    hf_key = os.getenv("HF_API_KEY", "")

uploaded_pdf = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_pdf:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_pdf.read())
        tmp.flush()
        pdf_path = tmp.name

    st.info("⏳ Extracting text and building vector index… This may take a few seconds.")

    try:
        emb_client = EmbeddingClient()   # FIXED HERE
        index, chunks = build_rag_from_pdf(pdf_path, emb_client)

        st.success("✅ RAG index built successfully! You can now ask questions.")

        st.session_state["index"] = index
        st.session_state["emb_client"] = emb_client

    except Exception as e:
        st.error(f"Error while building index: {e}")

if "index" in st.session_state:
    st.subheader("❓ Ask a question about the PDF")

    question = st.text_input("Enter your question:")

    top_k = st.slider("Number of context chunks:", 1, 8, 4)

    if st.button("Get Answer"):
        contexts, scores = query_rag(
            st.session_state["index"],
            st.session_state["emb_client"],
            question,
            top_k=top_k
        )

        st.markdown("### 🔍 Retrieved Contexts")
        for i, (ctx, score) in enumerate(zip(contexts, scores), start=1):
            st.markdown(f"**Chunk {i}** (Score: {score:.3f})")
            st.write(ctx)
            st.write("---")

        if not hf_key:
            st.warning("⚠ Please enter your Hugging Face API key to generate answers.")
        else:
            answer = generate_answer_with_hf(hf_key, question, contexts)
            st.markdown("### 🧠 AI Answer")
            st.write(answer)
