# rag_engine.py
"""
RAG engine: PDF text extraction, chunking, embeddings (HF or local), FAISS index build & query,
and LLM answer generation using Hugging Face Inference API.
"""

import os
import json
from typing import List, Tuple, Optional
import pdfplumber
import numpy as np
import requests

# Try loading local embedding model
try:
    from sentence_transformers import SentenceTransformer
    _HAS_SMODEL = True
except Exception:
    _HAS_SMODEL = False

# Try loading FAISS
try:
    import faiss
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False


# ----------------------------
# MODEL CONFIG
# ----------------------------
HF_API_URL = "https://api-inference.huggingface.co"

# Working embedding model (supports new HF embedding API)
HF_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Generation model
HF_GEN_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"


# ----------------------------
# PDF TEXT EXTRACTION
# ----------------------------
def extract_text_from_pdf(path: str) -> str:
    texts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            txt = page.extract_text()
            if txt:
                texts.append(txt)
    return "\n\n".join(texts)


# ----------------------------
# CHUNKING
# ----------------------------
def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150):
    chunks = []
    clean = text.replace("\r", " ")
    length = len(clean)
    start = 0

    while start < length:
        end = start + chunk_size
        chunk = clean[start:end]
        chunks.append((chunk, start, min(end, length)))
        start = end - overlap

    return chunks


# ----------------------------
# EMBEDDING CLIENT
# ----------------------------
class EmbeddingClient:
    def __init__(self, hf_api_key: Optional[str] = None, local_model_name="all-MiniLM-L6-v2"):
        self.hf_key = hf_api_key or os.getenv("HF_API_KEY")
        self.local = None

        if not self.hf_key and _HAS_SMODEL:
            try:
                self.local = SentenceTransformer(local_model_name)
            except Exception:
                self.local = None

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        if self.hf_key:
            return self._embed_hf(texts)
        elif self.local is not None:
            return self.local.encode(texts, convert_to_numpy=True)
        else:
            raise RuntimeError("No embedding model available.")

    def _embed_hf(self, texts):
        url = f"{HF_API_URL}/models/{HF_EMBED_MODEL}"
        headers = {"Authorization": f"Bearer {self.hf_key}"}

        payload = {
            "inputs": texts,
            "parameters": {"truncate": True},
            "options": {"wait_for_model": True}
        }

        r = requests.post(url, headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()

        vectors = []
        for item in data:
            if isinstance(item, list):
                vectors.append(item[0])  # first token embedding
            elif isinstance(item, dict) and "embedding" in item:
                vectors.append(item["embedding"])
            else:
                raise ValueError("Unexpected embedding output.")

        return np.array(vectors).astype("float32")


# ----------------------------
# FAISS VECTOR INDEX
# ----------------------------
class FaissIndex:
    def __init__(self, dim: int):
        if not _HAS_FAISS:
            raise RuntimeError("FAISS not installed.")
        self.index = faiss.IndexFlatIP(dim)
        self.metadatas = []

    def add(self, vectors, metas):
        vectors = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)
        self.index.add(vectors.astype("float32"))
        self.metadatas.extend(metas)

    def search(self, query_vec, top_k=5):
        q = query_vec / (np.linalg.norm(query_vec) + 1e-8)
        scores, idxs = self.index.search(q.reshape(1, -1), top_k)

        results = []
        for idx, score in zip(idxs[0], scores[0]):
            if idx >= 0:
                results.append((self.metadatas[idx], float(score)))
        return results


# ----------------------------
# BUILD RAG INDEX
# ----------------------------
def build_rag_from_pdf(pdf_path, emb_client):
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text, chunk_size=1000, overlap=200)
    chunk_texts = [c[0] for c in chunks]

    embeddings = emb_client.embed_batch(chunk_texts)
    dim = embeddings.shape[1]

    index = FaissIndex(dim)
    metas = [{"text": c[0], "start": c[1], "end": c[2]} for c in chunks]

    index.add(embeddings, metas)
    return index, chunks


# ----------------------------
# QUERY RAG
# ----------------------------
def query_rag(index, emb_client, question, top_k=4):
    q_emb = emb_client.embed_batch([question])[0]
    hits = index.search(q_emb, top_k)
    contexts = [h[0]["text"] for h in hits]
    scores = [h[1] for h in hits]
    return contexts, scores


# ----------------------------
# GENERATE ANSWER USING HF LLM
# ----------------------------
def generate_answer_with_hf(hf_api_key, question, contexts, max_length=256):
    url = f"{HF_API_URL}/models/{HF_GEN_MODEL}"
    headers = {"Authorization": f"Bearer {hf_api_key}"}

    context_block = "\n\n---\n\n".join(contexts)

    prompt = (
        "Use ONLY the context below to answer the question.\n"
        "If answer is not in the context, say 'I don't know.'\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\nAnswer:"
    )

    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": max_length, "temperature": 0.1}
    }

    r = requests.post(url, headers=headers, json=payload)
    r.raise_for_status()
    data = r.json()

    if isinstance(data, list) and "generated_text" in data[0]:
        return data[0]["generated_text"]

    return str(data)
