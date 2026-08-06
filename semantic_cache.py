import os
import sqlite3
import json
import numpy as np
from dotenv import load_dotenv

load_dotenv()
DEFAULT_DB_PATH = os.getenv("SEMANTIC_CACHE_DB_PATH", "semantic_cache.db")
DEFAULT_THRESHOLD = float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.92"))

class SemanticCache:
    def __init__(self, db_path=None, threshold=None):
        """
        Initialize the SQLite-based semantic cache.
        """
        self.db_path = db_path if db_path is not None else DEFAULT_DB_PATH
        self.threshold = threshold if threshold is not None else DEFAULT_THRESHOLD
        self._init_db()

    def _init_db(self):
        """Creates the cache table if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS query_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                embedding TEXT NOT NULL,
                response TEXT NOT NULL,
                sources TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()

    def get(self, query_text, query_embedding):
        """
        Checks the cache for a semantically similar query.
        
        Args:
            query_text (str): The raw text of the query.
            query_embedding (list or np.ndarray): The vector embedding of the query.
            
        Returns:
            tuple: (response, sources) if found, else None.
        """
        if query_embedding is None:
            return None
            
        query_vector = np.array(query_embedding, dtype=np.float32)
        norm_query = np.linalg.norm(query_vector)
        if norm_query == 0:
            return None

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT query, embedding, response, sources FROM query_cache")
        rows = cursor.fetchall()
        conn.close()

        best_score = -1.0
        best_match = None

        for cached_query, cached_emb_str, response, sources_str in rows:
            try:
                cached_vector = np.array(json.loads(cached_emb_str), dtype=np.float32)
            except Exception:
                continue
                
            norm_cached = np.linalg.norm(cached_vector)
            if norm_cached == 0:
                continue
                
            # Cosine similarity
            similarity = np.dot(query_vector, cached_vector) / (norm_query * norm_cached)
            
            if similarity > best_score:
                best_score = similarity
                best_match = (response, json.loads(sources_str), cached_query, float(similarity))

        if best_match and best_score >= self.threshold:
            # Found a semantic match!
            response, sources, matched_query, score = best_match
            print(f"[Semantic Cache Hit] Query: '{query_text}' matched '{matched_query}' with similarity {score:.4f}")
            return response, sources, matched_query, score
            
        return None

    def set(self, query_text, query_embedding, response, sources):
        """
        Saves a query, its embedding, response, and sources to the cache.
        
        Args:
            query_text (str): The raw text of the query.
            query_embedding (list): The vector embedding of the query.
            response (str): The generated answer.
            sources (list or set): The retrieved sources.
        """
        if query_embedding is None or not response:
            return
        emb_list = [float(x) for x in query_embedding]
        emb_str = json.dumps(emb_list)
        sources_str = json.dumps(list(sources))
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Avoid caching exact duplicates repeatedly
        cursor.execute("SELECT id FROM query_cache WHERE query = ?", (query_text,))
        exists = cursor.fetchone()
        
        if not exists:
            cursor.execute(
                "INSERT INTO query_cache (query, embedding, response, sources) VALUES (?, ?, ?, ?)",
                (query_text, emb_str, response, sources_str)
            )
            conn.commit()
            
        conn.close()
