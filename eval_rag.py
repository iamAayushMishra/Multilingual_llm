import os
import sys
import codecs
import time
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai
from dotenv import load_dotenv
from langdetect import detect

# Reconfigure stdout/stderr for Windows consoles to support Hindi printing
if sys.stdout.encoding != 'utf-8':
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'replace')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'replace')

from retrieval_pipeline import AdvancedRetrievalPipeline
from semantic_cache import SemanticCache

# Load environment
load_dotenv()
try:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
except AttributeError:
    print("Error: GOOGLE_API_KEY not found.")
    exit()

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "ncert_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "ncert_books")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "paraphrase-multilingual-MiniLM-L12-v2")
CROSS_ENCODER_MODEL_NAME = os.getenv("CROSS_ENCODER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-pro")
SEMANTIC_CACHE_DB_PATH = os.getenv("SEMANTIC_CACHE_DB_PATH", "semantic_cache.db")
SEMANTIC_CACHE_THRESHOLD = float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.92"))
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "4"))

# 5 Test Questions mapping English and Hindi queries to check retrieval and cross-lingual capability
TEST_QUERIES = [
    {
        "query": "Where did people first start growing wheat and barley?",
        "expected_topics": ["सुलेमान", "किरथर", "sulaiman", "kirthar"]
    },
    {
        "query": "नर्मदा नदी के किनारे रहने वाले लोग क्या काम करते थे?",
        "expected_topics": ["संग्राहक", "शिकारी", "hunter", "gatherer"]
    },
    {
        "query": "What are manuscripts and where were they written?",
        "expected_topics": ["पांडुलिपि", "ताड़पत्र", "भोजपत्र", "manuscript", "palm leaf"]
    },
    {
        "query": "सिंधु नदी के किनारे नगरों का विकास कब हुआ?",
        "expected_topics": ["4700", "forty-seven hundred"]
    },
    {
        "query": "Where was rice first grown?",
        "expected_topics": ["विंध्य", "vindhyas", "चावल", "rice"]
    }
]

def run_evaluation():
    print("Initializing components for evaluation...")
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
        print("Error: Collection not found. Please index documents first by running ingest.py.")
        return

    retriever = AdvancedRetrievalPipeline(
        chroma_collection=collection,
        cross_encoder_model_name=CROSS_ENCODER_MODEL_NAME
    )
    cache = SemanticCache(
        db_path=SEMANTIC_CACHE_DB_PATH,
        threshold=SEMANTIC_CACHE_THRESHOLD
    )
    model = genai.GenerativeModel(GEMINI_MODEL_NAME)

    print("\n" + "="*60)
    print("RUNNING NCERT-MITRA PIPELINE EVALUATION")
    print("="*60)

    results = []

    for i, test in enumerate(TEST_QUERIES):
        query = test["query"]
        print(f"\n[{i+1}/{len(TEST_QUERIES)}] Query: '{query}'")
        
        start_time = time.time()
        
        # 1. Detect language
        try:
            lang = detect(query)
        except Exception:
            lang = "en"
            
        # 2. Check Cache
        query_embedding = embedding_function([query])[0]
        cache_hit = cache.get(query, query_embedding)
        
        cached = False
        matched_q = None
        sim = 0.0
        
        if cache_hit:
            answer, sources, matched_q, sim = cache_hit
            cached = True
            latency = (time.time() - start_time) * 1000
            print(f"   -> [CACHE HIT] Matched: '{matched_q}' (Similarity: {sim:.4f})")
        else:
            # Cache miss, run full RAG pipeline
            # 3. Translate query if not Hindi
            if lang != "hi":
                print("   -> [Translation] Translating English query to Hindi search terms...")
                trans_prompt = f"Translate the following search query into search keywords/phrases in Hindi (without extra text or explanations):\nQuery: {query}"
                try:
                    search_query = model.generate_content(trans_prompt).text.strip()
                    print(f"      Translated: '{search_query}'")
                except Exception:
                    search_query = query
            else:
                search_query = query
                
            # 4. Retrieval & Re-ranking
            print("   -> [Retrieval] Performing hybrid search and cross-encoder re-ranking...")
            retrieved, sources = retriever.retrieve_hybrid_and_rerank(search_query, top_k=RETRIEVAL_TOP_K)
            
            # 5. Generation
            print("   -> [LLM Generation] Generating response via Gemini Pro...")
            context = "\n\n".join(retrieved)
            prompt = f"""
            You are a helpful AI assistant for students named "NCERT-Mitra".
            Your task is to answer the user's question based ONLY on the context provided below.
            
            CRITICAL RULES:
            1. Answer the question in the SAME language that the user asked it. If they asked in English, answer in English. If they asked in Hindi, answer in Hindi.
            2. If the context does not contain the answer, you MUST say "I am sorry, but I cannot find the answer to that question in the provided material." in the user's language.
            3. Do not use any external knowledge.

            CONTEXT (in Hindi):
            ---
            {context}
            ---

            USER'S QUESTION:
            {query}

            YOUR ANSWER:
            """
            success = False
            try:
                answer = model.generate_content(prompt).text.strip()
                success = True
            except Exception as e:
                answer = f"Error: {e}"
                
            # 6. Save to cache only on success
            if success:
                cache.set(query, query_embedding, answer, sources)
                print("   -> [Cache Saved] Response cached successfully.")
            else:
                print("   -> [Cache Bypassed] Generation failed, response not cached.")
            latency = (time.time() - start_time) * 1000

        print(f"   -> Latency: {latency:.2f} ms")
        print(f"   -> Sources: {sources}")
        print(f"   -> Answer snippet: {answer[:120]}...")
        
        # Simple topic check to measure correctness
        passed = any(topic.lower() in answer.lower() or topic.lower() in "".join(retrieved).lower() for topic in test["expected_topics"])
        print(f"   -> Ground Truth Match: {'PASS' if passed else 'FAIL'}")
        
        results.append({
            "query": query,
            "latency_ms": latency,
            "cache_hit": cached,
            "passed": passed
        })

    print("\n" + "="*60)
    print("EVALUATION RUN SUMMARY")
    print("="*60)
    total_latency = sum(r["latency_ms"] for r in results)
    avg_latency = total_latency / len(results)
    cache_hits = sum(1 for r in results if r["cache_hit"])
    passed_runs = sum(1 for r in results if r["passed"])
    
    print(f"Total Queries: {len(results)}")
    print(f"Average Latency: {avg_latency:.2f} ms")
    print(f"Cache Hit Rate: {cache_hits / len(results):.1%}")
    print(f"Context/Answer Accuracy Rate: {passed_runs / len(results):.1%}")
    print("="*60)

if __name__ == "__main__":
    run_evaluation()
