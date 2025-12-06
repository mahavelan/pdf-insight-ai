import streamlit as st
import tempfile
from rag_engine import (
    EmbeddingClient,
    build_rag_from_pdf,
    query_rag,
    generate_answer_via_space,
)

st.set_page_config(page_title="PDF Insight AI", layout="wide")

st.title("📄 PDF Insight AI — Intelligent Q&A System")

st.write("""
Upload any PDF and ask questions.

**Pipeline:**
- 🔍 Local MiniLM embeddings + FAISS for retrieval  
- 🤖 Custom Hugging Face Space (Phi-2) for answer generation  
""")

uploaded_pdf = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_pdf:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_pdf.read())
        tmp.flush()
        pdf_path = tmp.name

    st.info("⏳ Extracting text and building vector index…")

    try:
        emb_client = EmbeddingClient()
        index, chunks = build_rag_from_pdf(pdf_path, emb_client)

        st.success("✅ Index built successfully! You can now ask questions.")
        st.session_state["index"] = index
        st.session_state["emb_client"] = emb_client

    except Exception as e:
        st.error(f"Error while building index: {e}")

if "index" in st.session_state:
    st.subheader("❓ Ask a Question about the PDF")

    question = st.text_input("Enter your question:")
    top_k = st.slider("Number of context chunks to retrieve:", 1, 8, 4)

    if st.button("Get Answer") and question.strip():
        contexts, scores = query_rag(
            st.session_state["index"],
            st.session_state["emb_client"],
            question,
            top_k=top_k,
        )

        st.markdown("### 🧩 Retrieved Contexts")
        for i, (ctx, score) in enumerate(zip(contexts, scores), start=1):
            st.markdown(f"**Chunk {i}** (score: {score:.3f})")
            st.write(ctx)
            st.write("---")

        st.markdown("### 🤖 AI Answer (from your HF Space)")
        try:
            answer = generate_answer_via_space(question, contexts)
            st.write(answer)
        except Exception as e:
            st.error(f"Error while calling LLM Space: {e}")
