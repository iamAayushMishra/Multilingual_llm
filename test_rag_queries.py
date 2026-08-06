import os
import sys
import codecs
import time
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai
from dotenv import load_dotenv

# Reconfigure stdout/stderr for Windows console to handle Hindi encoding
if sys.stdout.encoding != 'utf-8':
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'replace')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'replace')

from retrieval_pipeline import AdvancedRetrievalPipeline
from semantic_cache import SemanticCache

# Load environment configuration variables
load_dotenv()

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "ncert_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "ncert_books")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "paraphrase-multilingual-MiniLM-L12-v2")
CROSS_ENCODER_MODEL_NAME = os.getenv("CROSS_ENCODER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-pro")
SEMANTIC_CACHE_DB_PATH = os.getenv("SEMANTIC_CACHE_DB_PATH", "semantic_cache.db")
SEMANTIC_CACHE_THRESHOLD = float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.92"))
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "4"))

TEST_CASES = [
    {
        "id": 1,
        "title": "Donkey as a Fool vs. Highest Virtues",
        "query": "Why is the donkey considered a fool by human standards, and how does the author challenge this assumption based on the text?"
    },
    {
        "id": 2,
        "title": "The Relationship Between Heera and Moti",
        "query": "How did the two oxen, Heera and Moti, express their mutual affection and deep friendship while working or resting?"
    },
    {
        "id": 3,
        "title": "Escape from Jhuri's In-Laws (Gaya's House)",
        "query": "Why did Heera and Moti refuse to eat their food when they first arrived at Gaya's (Jhuri's brother-in-law) house, and how did they escape that night?"
    }
]

def main():
    api_key = os.getenv("GOOGLE_API_KEY")
    has_valid_key = api_key and api_key != "YOUR_API_KEY_HERE"
    
    if has_valid_key:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(GEMINI_MODEL_NAME)
        print("API Key loaded successfully. Full RAG pipeline (Retrieval + Translation + Generation) is active.")
    else:
        model = None
        print("GOOGLE_API_KEY is not set or has placeholder value. Bypassing Gemini API generation.")
        print("Running local MULTILINGUAL RETRIEVAL & CACHING verification only.")

    # Initialize ChromaDB and Local Pipeline
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME
    )
    db_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    
    try:
        collection = db_client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_function
        )
    except Exception:
        print(f"Error: Collection '{COLLECTION_NAME}' not found. Please run ingest.py first.")
        return

    retriever = AdvancedRetrievalPipeline(collection, cross_encoder_model_name=CROSS_ENCODER_MODEL_NAME)
    cache = SemanticCache(db_path=SEMANTIC_CACHE_DB_PATH, threshold=SEMANTIC_CACHE_THRESHOLD)

    # We will run the evaluation TWICE to demonstrate caching in action!
    for run in [1, 2]:
        print("\n" + "="*80)
        print(f"RUN {run}: TESTING INPUTS & RETRIEVAL PIPELINE")
        print("="*80)
        
        for case in TEST_CASES:
            print(f"\n[Test Case {case['id']}] {case['title']}")
            print(f"Question: \"{case['query']}\"")
            
            start_time = time.time()
            
            # A. Check Cache
            query_embedding = embedding_function([case["query"]])[0]
            cached_hit = cache.get(case["query"], query_embedding)
            
            if cached_hit:
                ans, src, matched_q, sim = cached_hit
                latency = (time.time() - start_time) * 1000
                print(f"--> [CACHE HIT] Similarity: {sim:.4f} against match: '{matched_q}'")
                print(f"--> Latency: {latency:.2f} ms")
                print(f"--> Answer:\n{ans}")
                print(f"--> Sources: {src}")
                print("-"*80)
                continue
                
            # B. If Cache Miss, run pipeline
            # 1. Translate Query to Hindi search terms for sparse matching if API is available
            search_query = case["query"]
            if has_valid_key:
                print("--> [Translation] Translating English query to Hindi keywords for database search...")
                prompt = f"Translate the following search query into search keywords/phrases in Hindi (without extra text or explanations):\nQuery: {case['query']}"
                try:
                    search_query = model.generate_content(prompt).text.strip()
                    print(f"    Translated Keywords: '{search_query}'")
                except Exception as e:
                    print(f"    Translation failed: {e}. Using original query.")
            
            # 2. Local Multilingual Hybrid Search & Re-ranking
            print("--> [Retrieval] Searching database (Vector + BM25) and re-ranking candidates...")
            retrieved, sources = retriever.retrieve_hybrid_and_rerank(search_query, top_k=RETRIEVAL_TOP_K)
            
            print(f"--> [Retrieval Complete] Found {len(retrieved)} relevant chunks.")
            print(f"--> Sources found: {sources}")
            
            # Print a snippet of the top retrieved context chunk
            if retrieved:
                print("--> Top Retrieved Context Snippet (Hindi):")
                # Clean up multiple newlines for cleaner print
                snippet = retrieved[0].replace('\n', ' ')
                # Show safe ascii escape for Windows console
                print(f"    \"{snippet[:200]}...\"")
            
            # 3. LLM Generation (if API key available)
            if has_valid_key:
                print("--> [Generation] Calling Gemini to generate answer...")
                context = "\n\n".join(retrieved)
                prompt = f"""
                You are a helpful AI assistant for students named "NCERT-Mitra".
                Your task is to answer the user's question based ONLY on the context provided below.
                
                CRITICAL RULES:
                1. Answer the question in the SAME language that the user asked it (English).
                2. If the context does not contain the answer, you MUST say "I am sorry, but I cannot find the answer to that question in the provided material."
                3. Do not use any external knowledge.

                CONTEXT (in Hindi):
                ---
                {context}
                ---

                USER'S QUESTION:
                {case['query']}

                YOUR ANSWER:
                """
                try:
                    ans = model.generate_content(prompt).text.strip()
                    print(f"--> Latency: {(time.time() - start_time)*1000:.2f} ms")
                    print(f"--> Generated Answer:\n{ans}")
                    # Cache the result
                    cache.set(case["query"], query_embedding, ans, sources)
                except Exception as e:
                    print(f"--> Generation failed: {e}")
            else:
                latency = (time.time() - start_time) * 1000
                print(f"--> Latency (Retrieval Only): {latency:.2f} ms")
                print("--> [Generation Bypassed] (Install a valid GOOGLE_API_KEY in .env to generate final LLM answers.)")
                
            print("-"*80)

if __name__ == "__main__":
    main()
