import os
import sys
import codecs
import time
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, Dict, Any

# Ensure UTF-8 output on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'replace')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'replace')

import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai
from dotenv import load_dotenv
from langdetect import detect

# Import our modular retrieval and cache components
from retrieval_pipeline import AdvancedRetrievalPipeline
from semantic_cache import SemanticCache

load_dotenv()

# --- Configurations ---
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "ncert_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "ncert_books")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "paraphrase-multilingual-MiniLM-L12-v2")
CROSS_ENCODER_MODEL_NAME = os.getenv("CROSS_ENCODER_MODEL_NAME", "BAAI/bge-reranker-base")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-3.5-flash")
SEMANTIC_CACHE_DB_PATH = os.getenv("SEMANTIC_CACHE_DB_PATH", "semantic_cache.db")
SEMANTIC_CACHE_THRESHOLD = float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.92"))
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "4"))

# --- Initialize API Client ---
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key or api_key == "YOUR_API_KEY_HERE":
    print("Warning: GOOGLE_API_KEY is not configured. Live RAG generation will be disabled.")
else:
    genai.configure(api_key=api_key)

# --- Initialize Pipelines (Load model weights once) ---
print("Initializing embedding function...")
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBEDDING_MODEL_NAME
)

print("Connecting to ChromaDB...")
db_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
try:
    collection = db_client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function
    )
except Exception as e:
    print(f"Error: Collection '{COLLECTION_NAME}' not found. Make sure you run ingest.py first! Details: {e}")
    sys.exit(1)

print("Initializing Advanced Retrieval Pipeline (BM25 + Dense + Reranking)...")
retriever = AdvancedRetrievalPipeline(collection, cross_encoder_model_name=CROSS_ENCODER_MODEL_NAME)

print("Initializing Semantic Cache...")
cache = SemanticCache(db_path=SEMANTIC_CACHE_DB_PATH, threshold=SEMANTIC_CACHE_THRESHOLD)

# --- FastAPI Setup ---
app = FastAPI(title="NCERT-Mitra Custom API", version="1.0")

# Enable CORS for local developments
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Request/Response Models ---
class QueryRequest(BaseModel):
    query: str
    filters: Optional[Dict[str, Any]] = None

class QueryResponse(BaseModel):
    answer: str
    sources: list
    metadata: dict

# --- Translation & Language Detection Helpers ---
def detect_language(text: str) -> str:
    try:
        return detect(text)
    except Exception:
        return "en"

def translate_query_if_needed(query_text: str, source_lang: str, target_lang: str = "hi") -> str:
    if source_lang == target_lang or not api_key or api_key == "YOUR_API_KEY_HERE":
        return query_text
    
    prompt = f"Translate the following search query into a simplified set of search keywords/phrases in Hindi (without any introduction, greeting, or explanation):\nQuery: {query_text}"
    try:
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        response = model.generate_content(prompt)
        translated = response.text.strip()
        print(f"   -> Translated English query to Hindi search terms: '{translated}'")
        return translated
    except Exception as e:
        print(f"   -> Translation failed: {e}. Using original query.")
        return query_text

# --- API Endpoints ---

@app.get("/api/metadata")
def get_index_metadata():
    """Fetches unique subjects and classes available in ChromaDB to populate filters."""
    try:
        db_data = collection.get(include=["metadatas"])
        metadatas = db_data.get("metadatas", [])
        
        subjects = sorted(list(set(m.get("subject") for m in metadatas if m and m.get("subject"))))
        classes = sorted(list(set(m.get("class") for m in metadatas if m and m.get("class"))))
        
        return {"subjects": subjects, "classes": classes}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch metadata: {e}")

@app.post("/api/query", response_model=QueryResponse)
def execute_query(request: QueryRequest):
    """Executes the custom RAG query flow (Cache Check -> Search -> Re-rank -> LLM generation)."""
    start_time = time.time()
    
    user_query = request.query.strip()
    filters = request.filters
    
    # Filter out empty string filters from the request
    if filters:
        filters = {k: v for k, v in filters.items() if v}
        if not filters:
            filters = None

    if not user_query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    # 1. Semantic Cache Lookup
    query_embedding = embedding_function([user_query])[0]
    cached_hit = cache.get(user_query, query_embedding)
    
    if cached_hit:
        answer, sources, matched_query, similarity = cached_hit
        latency = (time.time() - start_time) * 1000
        return QueryResponse(
            answer=answer,
            sources=sources,
            metadata={
                "is_cached": True,
                "matched_query": matched_query,
                "similarity": round(similarity, 4),
                "latency_ms": round(latency, 2)
            }
        )

    # 2. Pre-retrieval language processing
    user_lang = detect_language(user_query)
    search_query = translate_query_if_needed(user_query, user_lang, target_lang="hi")

    # 3. Hybrid Search & Re-ranking (with metadata filters)
    retrieved_documents, sources = retriever.retrieve_hybrid_and_rerank(
        query=search_query,
        top_k=RETRIEVAL_TOP_K,
        filters=filters
    )

    if not retrieved_documents:
        no_doc_msg = (
            "I am sorry, but I cannot find any relevant context in the textbooks to answer your question."
            if user_lang != "hi"
            else "मुझे क्षमा करें, मैं आपकी प्रश्न का उत्तर देने के लिए पाठ्यपुस्तकों में कोई प्रासंगिक संदर्भ नहीं पा रहा हूँ।"
        )
        latency = (time.time() - start_time) * 1000
        return QueryResponse(
            answer=no_doc_msg,
            sources=[],
            metadata={
                "is_cached": False,
                "latency_ms": round(latency, 2)
            }
        )

    # 4. Generate Answer via Gemini
    if not api_key or api_key == "YOUR_API_KEY_HERE":
        # API key not set, return retrieved context snippet to help them debug
        context_preview = "\n\n".join(retrieved_documents[:2])
        no_key_msg = (
            f"API Key is missing or invalid. Set GOOGLE_API_KEY in .env to enable LLM generation.\n\n"
            f"**Retrieved Context Snippets (Hindi):**\n---\n{context_preview}"
        )
        latency = (time.time() - start_time) * 1000
        return QueryResponse(
            answer=no_key_msg,
            sources=sources,
            metadata={
                "is_cached": False,
                "latency_ms": round(latency, 2)
            }
        )

    context = "\n\n".join(retrieved_documents)
    prompt_template = f"""
    You are a helpful AI assistant for students named "NCERT-Mitra".
    Your task is to answer the user's question based ONLY on the context provided below.
    
    CRITICAL RULES:
    1. Answer the question in the SAME language that the user asked it. If they asked in English, answer in English. If they asked in Hindi, answer in Hindi.
    2. If the context does not contain the answer, you MUST say "I am sorry, but I cannot find the answer to that question in the provided material." in the user's language.
    3. Do not use any external knowledge. Be concise, clear, and informative.

    CONTEXT (in Hindi):
    ---
    {context}
    ---

    USER'S QUESTION:
    {user_query}

    YOUR ANSWER:
    """

    success = False
    try:
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        response = model.generate_content(prompt_template)
        answer = response.text.strip()
        success = True
    except Exception as e:
        answer = f"An error occurred while generating the answer: {e}"

    # 5. Cache the result (only if generation succeeded)
    if success:
        cache.set(user_query, query_embedding, answer, sources)

    latency = (time.time() - start_time) * 1000
    return QueryResponse(
        answer=answer,
        sources=sources,
        metadata={
            "is_cached": False,
            "latency_ms": round(latency, 2)
        }
    )

# --- Serve Static UI Files ---
# Create static directory if not exists
os.makedirs("static", exist_ok=True)
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    print("\n" + "="*50)
    print("NCERT-Mitra API Server running at: http://localhost:8000")
    print("Open http://localhost:8000 in your browser to view custom UI.")
    print("="*50 + "\n")
    uvicorn.run(app, host="127.0.0.1", port=8000)
