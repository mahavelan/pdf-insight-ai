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

st.title("📄 PDF Insight AI — Intelligent Q&A System")

st.write("""
Upload any PDF and ask questions.  
Using **RAG (Retrieval-Augmented Generation)** powered by:
- **MPNet embeddings** (high accuracy)
- **Mistral-7B-Instruct** (powerful LLM)
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

    st.info("⏳ Extracting text and building vector index… This may take 20–40 seconds.")
    try:
        emb_client = EmbeddingClient(hf_api_key=hf_key)
        index, chunks = build_rag_from_pdf(pdf_path, emb_client)
        st.success("✅ Index built successfully! Ask your question below.")
        st.session_state["index"] = index
        st.session_state["emb_client"] = emb_client
    except Exception as e:
        st.error(f"Error while building index: {e}")

if "index" in st.session_state:
    st.subheader("❓ Ask a Question")
    q = st.text_input("Enter your question about the PDF:")

    top_k = st.slider("Number of context chunks:", 1, 8, 4)

    if st.button("Get Answer"):
        contexts, scores = query_rag(
            st.session_state["index"],
            st.session_state["emb_client"],
            q,
            top_k=top_k
        )

        st.markdown("### 🔍 Retrieved Contexts:")
        for i, (ctx, score) in enumerate(zip(contexts, scores), start=1):
            st.markdown(f"**Chunk {i}** (score: {score:.3f})")
            st.write(ctx)

        if not hf_key:
            st.warning("⚠ Provide HF API key to generate an AI answer.")
        else:
            answer = generate_answer_with_hf(hf_key, q, contexts)
            st.markdown("### 🧠 AI Answer")
            st.write(answer)
