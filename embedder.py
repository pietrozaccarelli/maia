import os
import pickle
import subprocess
import trafilatura
import faiss
import numpy as np
import re
import hashlib
import logging
from sentence_transformers import SentenceTransformer, CrossEncoder

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------
# CONFIG
# ------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
embeddings_dir =f"{current_dir}/embeddings"  # folder for all embeddings
os.makedirs(embeddings_dir, exist_ok=True)
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
reranker_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ------------------------
# HELPERS
# ------------------------
from time import sleep
import random
from google_search import google_it

import requests
from fake_useragent import UserAgent

from time import sleep
import random
import logging

logger = logging.getLogger(__name__)

def google_it_with_retry(query, max_retries=3, delay=2):
    for attempt in range(max_retries):
        try:
            result = google_it(query)
            if result and result.strip():
                return result
            else:
                logger.warning(f"Attempt {attempt + 1}: Empty result for query '{query}'. Retrying in {delay} seconds...")
                sleep(delay * (attempt + 1))
        except Exception as e:
            logger.error(f"Attempt {attempt + 1}: Error in google_it: {str(e)}. Retrying...")
            sleep(delay * (attempt + 1))
    logger.error(f"Failed to get results for query '{query}' after {max_retries} attempts.")
    return None



def text_to_slug(text: str) -> str:
    """Make a unique slug for storing embeddings based on text hash."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()

def get_paths_for_slug(slug: str):
    index_path = os.path.join(embeddings_dir, f"{slug}_index.bin")
    chunks_path = os.path.join(embeddings_dir, f"{slug}_chunks.pkl")
    return index_path, chunks_path

# ------------------------
# STEP 1: SENTENCE-BASED CHUNKING
# ------------------------
def chunk_text(text, max_words=200, overlap=50):
    if not text or not text.strip():
        logger.warning("Empty text received for chunking.")
        return []
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks, current, length = [], [], 0
    for s in sentences:
        words = s.split()
        if length + len(words) > max_words and current:
            chunks.append(" ".join(current))
            current = current[-overlap:] if overlap > 0 else []
            length = len(current)
        current.extend(words)
        length += len(words)
    if current:
        chunks.append(" ".join(current))
    return chunks

# ------------------------
# STEP 2: EMBEDDINGS
# ------------------------
embed_model = SentenceTransformer(embedding_model_name)

def embed_chunks(chunks):
    if not chunks:
        logger.warning("No chunks to embed.")
        return np.array([])
    embeddings = embed_model.encode(chunks, convert_to_numpy=True)
    if embeddings.ndim == 1:
        embeddings = embeddings[np.newaxis, :]
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings

# ------------------------
# STEP 3: PERSISTENCE
# ------------------------
def save_faiss_index(index, path):
    try:
        faiss.write_index(index, path)
    except Exception as e:
        logger.error(f"Error saving FAISS index: {str(e)}")

def load_faiss_index(path, expected_dim=None):
    if os.path.isfile(path):
        try:
            index = faiss.read_index(path)
            if expected_dim and index.d != expected_dim:
                logger.warning(f"Dimension mismatch for index {path}.")
                return None
            return index
        except Exception as e:
            logger.error(f"Error loading FAISS index: {str(e)}")
            return None
    return None

def save_text_chunks(chunks, embeddings, path, source_id):
    data = {
        "chunks": chunks,
        "embeddings": embeddings,
        "source_id": source_id,
        "model": embedding_model_name,
        "dim": embed_model.get_sentence_embedding_dimension(),
    }
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def load_text_chunks(path, source_id):
    if os.path.isfile(path):
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            if data.get("model") != embedding_model_name:
                logger.warning(f"Model mismatch for chunks {path}.")
                return None, None
            return data["chunks"], data["embeddings"]
        except Exception as e:
            logger.error(f"Error loading text chunks: {str(e)}")
            return None, None
    return None, None

# ------------------------
# STEP 4: BUILD FAISS INDEX
# ------------------------
def build_index(chunks):
    if not chunks:
        logger.warning("No chunks to build index.")
        return None, np.array([])
    embeddings = embed_chunks(chunks)
    if embeddings.shape[0] == 0:
        logger.warning("No embeddings created, cannot build FAISS index.")
        return None, np.array([])
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype('float32'))
    return index, embeddings

# ------------------------
# STEP 5: QUERY + RERANK (MULTI-INDEX)
# ------------------------
reranker = CrossEncoder(reranker_model_name)

def query_all_indexes(question, k=10, top_n=5):
    q_emb = embed_model.encode([question], convert_to_numpy=True)
    q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
    candidates = []
    for file in os.listdir(embeddings_dir):
        if file.endswith("_index.bin"):
            slug = file.replace("_index.bin", "")
            index_path = os.path.join(embeddings_dir, file)
            chunks_path = os.path.join(embeddings_dir, f"{slug}_chunks.pkl")
            index = load_faiss_index(index_path, expected_dim=embed_model.get_sentence_embedding_dimension())
            if index is None:
                logger.warning(f"Skipping index {index_path} due to dimension mismatch.")
                continue
            chunks, _ = load_text_chunks(chunks_path, source_id=slug)
            if chunks is None:
                logger.warning(f"Skipping chunks {chunks_path} due to model mismatch.")
                continue
            D, I = index.search(q_emb.astype('float32'), k)
            for i in I[0]:
                if 0 <= i < len(chunks):
                    candidates.append(chunks[i])
    if not candidates:
        logger.warning("No candidates found for query.")
        return []
    pairs = [(question, c) for c in candidates]
    scores = reranker.predict(pairs)
    reranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    best_chunks = [c for c, _ in reranked[:top_n]]
    return best_chunks

# ------------------------
# STEP 6: ASK GEMMA 3 VIA OLLAMA
# ------------------------
def ask_ollama(prompt, model="gemma3"):
    try:
        cmd = ["ollama", "run", model]
        result = subprocess.run(
            cmd,
            input=prompt,
            text=True,
            capture_output=True,
            encoding="utf-8",
            errors="replace"
        )
        return result.stdout.strip()
    except Exception as e:
        logger.error(f"Error asking Ollama: {str(e)}")
        return ""

# ------------------------
# MAIN FUNCTION (ROBUST)
# ------------------------

def inquire(question, num_links=5):
    logger.info(f"Preparing embeddings for question: {question}")
    try:
        google_text = google_it_with_retry(question)
        if not google_text:
            logger.error("Google search failed after retries.")
            return f"Please answer the following question directly:\n{question}"
        slug = text_to_slug(google_text)
        index_path, chunks_path = get_paths_for_slug(slug)
        if not (os.path.exists(index_path) and os.path.exists(chunks_path)):
            logger.info("Creating embeddings for Google search results...")
            chunks = chunk_text(google_text)
            if not chunks:
                logger.warning("No chunks generated from Google search results.")
                return f"Please answer the following question directly:\n{question}"
            index, embeddings = build_index(chunks)
            if index is None:
                logger.warning("Failed to build FAISS index.")
                return f"Please answer the following question directly:\n{question}"
            save_faiss_index(index, index_path)
            save_text_chunks(chunks, embeddings, chunks_path, slug)
        logger.info(f"Question: {question}")
        relevant_chunks = query_all_indexes(question, k=10, top_n=5)
        if not relevant_chunks:
            logger.warning("No relevant chunks found for question.")
            return f"Please answer the following question directly:\n{question}"
        context = "\n\n".join(relevant_chunks)
        prompt = f"""You're an expert professor answering questions.
Here's some context to help you answer the question:
{context}
Now answer precisely to the question:
{question}
"""
        return prompt
    except Exception as e:
        logger.error(f"Error in inquire: {str(e)}")
        return f"Please answer the following question directly:\n{question}"

# ------------------------
# UTILS: LOAD ALL EMBEDDINGS
# ------------------------
def load_all_embeddings_and_chunks(embeddings_dir):
    all_chunks = []
    all_embeddings = []
    index_map = {}
    for file in os.listdir(embeddings_dir):
        if file.endswith("_index.bin"):
            slug = file.replace("_index.bin", "")
            index_path = os.path.join(embeddings_dir, file)
            chunks_path = os.path.join(embeddings_dir, f"{slug}_chunks.pkl")
            index = faiss.read_index(index_path)
            chunks, embeddings = load_text_chunks(chunks_path, source_id=slug)
            if chunks is not None and embeddings is not None:
                all_chunks.extend(chunks)
                all_embeddings.append(embeddings)
                index_map[index] = len(all_chunks) - len(chunks)
    all_embeddings_np = np.concatenate(all_embeddings, axis=0) if all_embeddings else np.array([])
    return all_chunks, all_embeddings_np, index_map

def query_loaded_data(question, all_chunks, all_embeddings_np, reranker, k=10, top_n=5):
    q_emb = embed_model.encode([question], convert_to_numpy=True)
    q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
    if all_embeddings_np is None or all_embeddings_np.size == 0:
        return []
    dimension = all_embeddings_np.shape[1]
    if dimension <= 0:
        return []
    index = faiss.IndexFlatIP(dimension)
    index.add(all_embeddings_np.astype('float32'))
    D, I = index.search(q_emb.astype('float32'), k)
    candidates = []
    for i in I[0]:
        if 0 <= i < len(all_chunks):
            candidates.append(all_chunks[i])
    if not candidates:
        return []
    pairs = [(question, c) for c in candidates]
    scores = reranker.predict(pairs)
    reranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    best_chunks = [c for c, _ in reranked[:top_n]]
    return best_chunks

def ask_further_questions(question, embeddings_dir="embeddings", reranker_model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
    all_chunks, all_embeddings_np, index_map = load_all_embeddings_and_chunks(embeddings_dir)
    reranker = CrossEncoder(reranker_model_name)
    relevant_chunks = query_loaded_data(question, all_chunks, all_embeddings_np, reranker)
    context = "\n\n".join(relevant_chunks)
    if not context:
        return f"No relevant context available. Please answer directly:\n{question}"
    prompt = f"""You're an expert professor answering questions.
Here’s some context to help you answer the question:
{context}
Now answer precisely to the question:
{question}
"""
    return prompt
