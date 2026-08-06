import os
import sys
import codecs
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai
from dotenv import load_dotenv
from langdetect import detect, DetectorFactory

# Reconfigure stdout/stderr for Windows consoles to support Hindi printing
if sys.stdout.encoding != 'utf-8':
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'replace')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'replace')

# Ensure language detection is deterministic
DetectorFactory.seed = 0

from retrieval_pipeline import AdvancedRetrievalPipeline
from semantic_cache import SemanticCache

# --- CONFIGURATION ---
load_dotenv()

try:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
except AttributeError:
    print("Error: GOOGLE_API_KEY not found. Please ensure you have a .env file with the key.")
    exit()

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "ncert_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "ncert_books")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "paraphrase-multilingual-MiniLM-L12-v2")
CROSS_ENCODER_MODEL_NAME = os.getenv("CROSS_ENCODER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-pro")
SEMANTIC_CACHE_DB_PATH = os.getenv("SEMANTIC_CACHE_DB_PATH", "semantic_cache.db")
SEMANTIC_CACHE_THRESHOLD = float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.92"))
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "4"))


def detect_language(text):
    """Detects the language of the query. Defaults to 'en' on failure."""
    try:
        lang = detect(text)
        return lang
    except Exception:
        return "en"


def translate_query_if_needed(query_text, source_lang, target_lang="hi", model=None):
    """
    If the query is not in the target language (e.g. Hindi), translate it using Gemini
    so that exact keyword match (BM25) will work on the Hindi index.
    """
    if source_lang == target_lang:
        return query_text
        
    print(f"Translating query from '{source_lang}' to '{target_lang}' for optimal search...")
    prompt = f"Translate the following search query into search keywords/phrases in Hindi (without extra text or explanations):\nQuery: {query_text}"
    
    try:
        response = model.generate_content(prompt)
        translated = response.text.strip()
        print(f"Search Query Translation: '{translated}'")
        return translated
    except Exception as e:
        print(f"Translation error: {e}. Using original query.")
        return query_text


def main():
    # --- 1. Load ChromaDB & Models ---
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
    except Exception:
        print(f"Error: Collection '{COLLECTION_NAME}' not found. Please run ingest.py first.")
        return

    # Initialize Advanced Retrieval Pipeline and Semantic Cache
    retriever = AdvancedRetrievalPipeline(
        chroma_collection=collection,
        cross_encoder_model_name=CROSS_ENCODER_MODEL_NAME
    )
    cache = SemanticCache(
        db_path=SEMANTIC_CACHE_DB_PATH,
        threshold=SEMANTIC_CACHE_THRESHOLD
    )
    
    print("Initializing Gemini model...")
    model = genai.GenerativeModel(GEMINI_MODEL_NAME)
    
    print("\n--- NCERT-Mitra AI Assistant (Multilingual + Cache + Re-ranking) ---")
    print("Ask any question about your NCERT books in English or Hindi.")
    print("Type 'exit' to quit.\n")

    # --- 2. Q&A Loop ---
    while True:
        user_query = input("You: ")
        if user_query.lower() == 'exit':
            print("Thank you for using NCERT-Mitra. Goodbye!")
            break
        
        if not user_query.strip():
            continue
            
        # A. Language Detection
        user_lang = detect_language(user_query)
        print(f"[Language Detected]: {user_lang}")
        
        # B. Check Semantic Cache
        # We need the query's embedding to check the semantic cache
        query_embedding = embedding_function([user_query])[0]
        cached_hit = cache.get(user_query, query_embedding)
        
        if cached_hit:
            response, sources, matched_query, similarity = cached_hit
            print("\nNCERT-Mitra (FROM SEMANTIC CACHE):")
            print(response)
            print("\n--- Sources ---")
            for source in sources:
                print(f"- {source} (Cache Match: '{matched_query}' - Sim: {similarity:.2%})")
            print("\n" + "="*50 + "\n")
            continue

        # C. Query Preprocessing / Translation
        # Translate to Hindi if query is English, since target PDFs are Hindi ('ihga101' series)
        search_query = translate_query_if_needed(user_query, user_lang, "hi", model)
        
        # D. Advanced Hybrid Retrieval & Re-ranking
        print("Searching and re-ranking relevant contexts...")
        retrieved_documents, sources = retriever.retrieve_hybrid_and_rerank(
            query=search_query,
            top_k=RETRIEVAL_TOP_K
        )
        
        if not retrieved_documents:
            print("\nNCERT-Mitra:")
            print("I am sorry, but I cannot find any relevant documents to answer that question.")
            print("\n" + "="*50 + "\n")
            continue
            
        context = "\n\n".join(retrieved_documents)
        
        # E. Prompt Formulation (Cross-lingual support instruction)
        prompt_template = f"""
        You are a helpful AI assistant for students named "NCERT-Mitra".
        Your task is to answer the user's question based ONLY on the context provided below.
        
        CRITICAL RULES:
        1. Answer the question in the SAME language that the user asked it. If they asked in English, answer in English. If they asked in Hindi, answer in Hindi.
        2. If the context does not contain the answer, you MUST say "I am sorry, but I cannot find the answer to that question in the provided material." in the user's language.
        3. Do not use any external knowledge. Be concise and clear in your explanation.

        CONTEXT (in Hindi):
        ---
        {context}
        ---

        USER'S QUESTION (original):
        {user_query}

        YOUR ANSWER:
        """

        # F. Call Gemini
        print("Generating answer...")
        success = False
        try:
            response = model.generate_content(prompt_template)
            answer = response.text
            success = True
        except Exception as e:
            answer = f"An error occurred while generating the answer: {e}"

        # G. Save to Semantic Cache only on success
        if success:
            cache.set(user_query, query_embedding, answer, sources)

        # H. Display Output
        print("\nNCERT-Mitra:")
        print(answer)
        
        print("\n--- Sources ---")
        for source in sources:
            print(f"- {source}")
        print("\n" + "="*50 + "\n")


if __name__ == "__main__":
    main()