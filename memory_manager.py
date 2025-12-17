import os
import json
import sqlite3
import faiss
import pickle
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import logging

logger = logging.getLogger(__name__)

# --- Config ---
DB_FILE = "memory/facts.db"
FAISS_INDEX_FILE = "memory/vector_index.faiss"
TEXTS_FILE = "memory/memory_texts.pkl"
embedding_dim = 384
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# --- SQLite setup ---
os.makedirs(os.path.dirname(DB_FILE), exist_ok=True)
conn = sqlite3.connect(DB_FILE)
cur = conn.cursor()
cur.execute("""
CREATE TABLE IF NOT EXISTS facts (
    key TEXT PRIMARY KEY,
    value TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")
conn.commit()
conn.close()

# --- FAISS setup ---
if os.path.exists(FAISS_INDEX_FILE) and os.path.exists(TEXTS_FILE):
    index = faiss.read_index(FAISS_INDEX_FILE)
    memory_texts = pickle.load(open(TEXTS_FILE, "rb"))
else:
    index = faiss.IndexFlatL2(embedding_dim)
    memory_texts = []

# --- Fact functions ---
def store_fact(key, value):
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("INSERT OR REPLACE INTO facts (key, value) VALUES (?, ?)", (key, value))
    conn.commit()
    conn.close()
    logger.info(f"Stored fact: {key} = {value}")

def get_fact(key):
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("SELECT value FROM facts WHERE key=?", (key,))
    result = cur.fetchone()
    conn.close()
    return result[0] if result else None

def get_all_facts():
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("SELECT key, value FROM facts")
    results = cur.fetchall()
    conn.close()
    return {k: v for k, v in results}

# --- Episodic memory ---
def add_memory(text: str):
    if not text or not isinstance(text, str) or text.strip() == "":
        logger.warning("Skipping empty memory entry")
        return

    vector = embedder.encode([text])
    memory_texts.append(text)
    index.add(vector)
    logger.info(f"Added episodic memory: {text}")


def retrieve_memories(query, top_k=3):
    if not memory_texts:
        return []
    vector = embedder.encode([query])
    D, I = index.search(np.array(vector, dtype=np.float32), top_k)
    return [memory_texts[i] for i in I[0] if i < len(memory_texts)]

# --- Optional JSON fallback for full conversation history ---
def save_json_memory(messages, memory_file):
    os.makedirs(os.path.dirname(memory_file), exist_ok=True)
    messages_copy = []
    for m in messages:
        if hasattr(m, 'content') and m.content.strip():
            if isinstance(m, HumanMessage):
                role = "user"
            elif isinstance(m, AIMessage):
                role = "assistant"
            elif isinstance(m, SystemMessage):
                role = "system"
            else:
                continue
            # Debug: Print the content to check for invalid characters
            print(f"Debug - Message content: {m.content}")  # <-- Add this line
            messages_copy.append({"role": role, "content": m.content.splitlines()})
    with open(memory_file, "w", encoding="utf-8") as f:
        json.dump(messages_copy, f, ensure_ascii=False, indent=2)



