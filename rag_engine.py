# rag_engine.py
import os
import json
from typing import List, Tuple
import pdfplumber
import numpy as np
import requests

# SentenceTransformer local embeddings
from sentence_transformers import SentenceTransformer

# FAISS vector search
import faiss

HF_API_URL = "https://api-inference.huggingface.co"
HF_GEN_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"


# ----------------------------
# PDF TEXT EXTRACTION
# ----------------------------
def extract_text_from_pdf(path: str) -> str:
    texts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                texts.append(t)
    return "\n\n".join(texts)


# ----------------------------
# TEXT CHUNKING
# ----------------------------
def chunk_text(text, chunk_size=800, overlap=150):
    chunks = []
    text = text.replace("\r", " ")
    length = len(text)
    start = 0

    while start < length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append((chunk, start, min(end, length)))
        start = end - overlap

    return chunks


# ----------------------------
# LOCAL EMBEDDINGS (NO HF)
# ----------------------------
class EmbeddingClient:
    def __init__(self):
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def embed_batch(self, texts: List[str]):
        return self.model.encode(texts, convert_to_numpy=True)


# ----------------------------
# VECTOR INDEX (FAISS)
# ----------------------------
class FaissIndex:
    def __init__(self, dim):
        self.index = faiss.IndexFlatIP(dim)
        self.meta = []

    def add(self, vectors, metadatas):
        vectors = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)
        self.index.add(vectors.astype("float32"))
        self.meta.extend(metadatas)

    def search(self, vec, top_k=4):
        vec = vec / (np.linalg.norm(vec) + 1e-8)
        scores, idxs = self.index.search(vec.reshape(1, -1), top_k)
        results = []

        for idx, score in zip(idxs[0], scores[0]):
            if idx >= 0:
                results.append((self.meta[idx], float(score)))

        return results


# ----------------------------
# BUILD RAG INDEX
# ----------------------------
def build_rag_from_pdf(pdf_path, emb_client):
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)

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
# GENERATE ANSWER WITH HF LLM
# ----------------------------
def generate_answer_with_hf(hf_api_key, question, contexts, max_length=256):
    url = f"{HF_API_URL}/models/{HF_GEN_MODEL}"
    headers = {"Authorization": f"Bearer {hf_api_key}"}

    context_text = "\n\n---\n\n".join(contexts)

    prompt = (
        "You are an AI assistant. Answer ONLY using the context. "
        "If answer is not in the context, say 'I don't know.'\n\n"
        f"Context:\n{context_text}\n\n"
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
