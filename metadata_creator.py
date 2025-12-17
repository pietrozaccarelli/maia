import os
from datetime import datetime
import spacy
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

# Load spaCy model for entity recognition
try:
    nlp = spacy.load("en_core_web_sm")
except Exception:
    print("Spacy model not found. Run: python -m spacy download en_core_web_sm")
    nlp = spacy.blank("en")

# Load multilingual sentiment model
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"
embed_model = SentenceTransformer(EMBEDDER)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
topic_filepath = os.path.join(BASE_DIR, "topics.txt")

# Cache topics to prevent reading file on every request
_CACHED_TOPICS = []

def _load_topics(file_path: str) -> List[str]:
    """Loads topics with caching and error handling."""
    global _CACHED_TOPICS
    if _CACHED_TOPICS:
        return _CACHED_TOPICS
        
    if not os.path.exists(file_path):
        print(f"Warning: Topics file not found at {file_path}")
        # Return a default list if file is missing to prevent crashes
        return ["coding", "writing", "analysis", "general"]
        
    topics = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                topic = line.strip()
                if topic:
                    topics.append(topic)
        _CACHED_TOPICS = topics
    except Exception as e:
        print(f"Error reading topics file: {e}")
        return ["general"]
        
    return topics

def get_topic(prompt: str, topics_file: str, threshold: float = 0.2) -> str:
    """
    Detects topic using semantic similarity.
    Lowered default threshold slightly to be more responsive.
    """
    topics = _load_topics(topics_file)
    if not topics:
        return "general"

    # Encode prompt and topics
    # important: We only take the first 512 chars of prompt to avoid 
    # embedding noise if a huge text is accidentally passed
    sentences = [prompt[:512]] + topics
    embeddings = embed_model.encode(sentences, normalize_embeddings=True)

    prompt_vec = embeddings[0]
    topic_vecs = embeddings[1:]

    # Cosine similarity
    sims = np.dot(topic_vecs, prompt_vec)

    # Select best
    best_idx = int(np.argmax(sims))
    best_sim = float(sims[best_idx])
    best_topic = topics[best_idx]

    # Debug print to see what's happening
    print(f"Topic Detection: '{best_topic}' (Score: {best_sim:.2f})")

    if best_sim >= threshold:
        return best_topic
    else:
        return "general"

def get_sentiment(text):
    # Truncate text for BERT model to prevent errors on long inputs
    inputs = tokenizer(text[:512], return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    scores = torch.nn.functional.softmax(outputs.logits, dim=1)
    sentiment_score = torch.argmax(scores).item() + 1  # 1 to 5 stars
    sentiment_map = {
        1: "NEGATIVE",
        2: "NEGATIVE",
        3: "NEUTRAL",
        4: "POSITIVE",
        5: "POSITIVE"
    }
    return sentiment_map[sentiment_score], round(scores[0][sentiment_score - 1].item(), 2)

def enrich_prompt(prompt):
    timestamp = datetime.now().isoformat()
    role = "user"
    
    # NLP processing
    doc = nlp(prompt[:100000]) # Limit spaCy processing size
    entities = [ent.text for ent in doc.ents]
    
    # Sentiment analysis
    sentiment, sentiment_score = get_sentiment(prompt)
    
    # Intent detection
    intent = "question" if prompt.strip().endswith("?") else "statement"
    
    # Topic detection
    topic = get_topic(prompt=prompt, topics_file=topic_filepath)
    
    metadata = {
        "timestamp": timestamp,
        "role": role,
        "entities": entities,
        "sentiment": sentiment,
        "sentiment_score": sentiment_score,
        "intent": intent,
        "topic": topic,
        "text_length": len(prompt),
        "word_count": len(prompt.split())
    }
    
    return {
        "role": role,
        "content": prompt,
        "metadata": metadata
    }