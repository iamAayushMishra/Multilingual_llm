import os
import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai
from dotenv import load_dotenv
from langdetect import detect

from retrieval_pipeline import AdvancedRetrievalPipeline
from semantic_cache import SemanticCache

# --- CONFIGURATION ---
load_dotenv()

try:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
except AttributeError:
    st.error("Error: GOOGLE_API_KEY not found. Please ensure you have a .env file with the key.")
    st.stop()

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "ncert_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "ncert_books")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "paraphrase-multilingual-MiniLM-L12-v2")
CROSS_ENCODER_MODEL_NAME = os.getenv("CROSS_ENCODER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-pro")
SEMANTIC_CACHE_DB_PATH = os.getenv("SEMANTIC_CACHE_DB_PATH", "semantic_cache.db")
SEMANTIC_CACHE_THRESHOLD = float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.92"))
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "4"))

# --- CACHED FUNCTIONS ---
@st.cache_resource
def load_chroma_collection():
    """Loads the ChromaDB collection and the embedding function."""
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME
    )
    db_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    collection = db_client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function
    )
    return collection, embedding_function

@st.cache_resource
def load_retrieval_pipeline(_collection):
    """Loads the advanced hybrid retrieval + cross-encoder re-ranking pipeline."""
    return AdvancedRetrievalPipeline(_collection, cross_encoder_model_name=CROSS_ENCODER_MODEL_NAME)

@st.cache_resource
def load_semantic_cache():
    """Loads the local SQLite-based semantic cache."""
    return SemanticCache(db_path=SEMANTIC_CACHE_DB_PATH, threshold=SEMANTIC_CACHE_THRESHOLD)

@st.cache_resource
def load_gemini_model():
    """Loads the Gemini model."""
    return genai.GenerativeModel(GEMINI_MODEL_NAME)

def detect_language(text):
    """Detects query language."""
    try:
        return detect(text)
    except Exception:
        return "en"

def translate_query_if_needed(query_text, source_lang, target_lang="hi", model=None):
    """Translates query to target index language (Hindi) if needed."""
    if source_lang == target_lang:
        return query_text
    
    prompt = f"Translate the following search query into search keywords/phrases in Hindi (without extra text or explanations):\nQuery: {query_text}"
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception:
        return query_text

# --- PAGE CONFIG & CUSTOM STYLE INJECTION ---
st.set_page_config(page_title="NCERT-Mitra", page_icon="📚", layout="wide")

# Custom CSS for glassmorphic premium UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    /* Main Background & Fonts */
    .stApp {
        background: radial-gradient(circle at 50% 50%, #0f0c20 0%, #15102a 100%) !important;
        font-family: 'Outfit', sans-serif !important;
        color: #e2e8f0 !important;
    }
    
    /* Headers & Text */
    h1, h2, h3, p, span, label {
        font-family: 'Outfit', sans-serif !important;
    }
    
    h1 {
        background: linear-gradient(135deg, #a29bfe 0%, #6c5ce7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700 !important;
        text-shadow: 0px 4px 20px rgba(108, 92, 231, 0.1);
    }
    
    /* Glassmorphism sidebar */
    section[data-testid="stSidebar"] {
        background-color: rgba(15, 12, 32, 0.8) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(15px);
    }
    
    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.02);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(8px);
    }
    
    /* Cache Hit Badge Style */
    .cache-badge {
        background: linear-gradient(135deg, #00b894 0%, #00cec9 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        margin-bottom: 12px;
        box-shadow: 0 4px 12px rgba(0, 206, 201, 0.3);
        animation: pulse-glow 2s infinite;
    }
    
    @keyframes pulse-glow {
        0% { box-shadow: 0 0 0 0 rgba(0, 206, 201, 0.4); }
        70% { box-shadow: 0 0 0 8px rgba(0, 206, 201, 0); }
        100% { box-shadow: 0 0 0 0 rgba(0, 206, 201, 0); }
    }
    
    /* Custom Chat Bubbles */
    div[data-testid="stChatMessage"] {
        background-color: rgba(255, 255, 255, 0.015) !important;
        border: 1px solid rgba(255, 255, 255, 0.03) !important;
        border-radius: 12px !important;
        padding: 15px !important;
        margin-bottom: 10px !important;
        backdrop-filter: blur(2px);
    }
    
    /* Accent borders for chat roles */
    div[data-testid="stChatMessage"][data-test-user-role="user"] {
        border-left: 4px solid #a29bfe !important;
        background-color: rgba(162, 155, 254, 0.03) !important;
    }
    
    div[data-testid="stChatMessage"][data-test-user-role="assistant"] {
        border-left: 4px solid #6c5ce7 !important;
        background-color: rgba(108, 92, 231, 0.03) !important;
    }
    
    /* Streamlit widgets modifications */
    .stTextInput>div>div>input {
        background-color: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        color: #e2e8f0 !important;
        border-radius: 8px !important;
    }
    
    .stTextInput>div>div>input:focus {
        border-color: #a29bfe !important;
        box-shadow: 0 0 0 2px rgba(162, 155, 254, 0.2) !important;
    }
    
    /* Source pill styles */
    .source-pill {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 0.8rem;
        display: inline-block;
        margin-right: 6px;
        margin-top: 4px;
        color: #a29bfe;
    }
</style>
""", unsafe_allow_html=True)

# --- BACKEND LOGIC ---
def process_rag_flow(user_query, collection, embedding_function, retriever, cache, model, filters=None):
    """
    Handles RAG pipeline: Cache Check -> Language Detection -> Query Translation -> Hybrid Search & Re-ranking -> LLM Response.
    """
    user_lang = detect_language(user_query)
    
    # 1. Check SQLite Semantic Cache
    query_embedding = embedding_function([user_query])[0]
    cached_hit = cache.get(user_query, query_embedding)
    
    if cached_hit:
        answer, sources, matched_query, similarity = cached_hit
        return answer, sources, {"is_cached": True, "matched_query": matched_query, "similarity": similarity}
        
    # 2. Query Translation (if user asked in English, translate query to Hindi search terms for database lookup)
    search_query = translate_query_if_needed(user_query, user_lang, "hi", model)
    
    # 3. Retrieve and Re-rank (with metadata filters if selected)
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
        return no_doc_msg, [], {"is_cached": False}
        
    context = "\n\n".join(retrieved_documents)
    
    # 4. Prompt Engineering & Cross-lingual generation instruction
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
        response = model.generate_content(prompt_template)
        answer = response.text
        success = True
    except Exception as e:
        answer = f"An error occurred while generating the answer: {e}"
        
    # 5. Save generated response to semantic cache only on success
    if success:
        cache.set(user_query, query_embedding, answer, sources)
    
    return answer, sources, {"is_cached": False}

# --- STREAMLIT PAGE LAYOUT ---
# Header
col1, col2 = st.columns([0.1, 0.9])
with col1:
    st.markdown("<h1 style='text-align: center; font-size: 3rem; margin:0;'>📚</h1>", unsafe_allow_html=True)
with col2:
    st.title("NCERT-Mitra AI Assistant")
    st.caption("Advanced Multilingual Retrieval (Hybrid Search & Re-ranking + Semantic Caching)")

# Load components
try:
    collection, embedding_function = load_chroma_collection()
    retriever = load_retrieval_pipeline(collection)
    cache = load_semantic_cache()
    model = load_gemini_model()
    db_loaded = True
except Exception as e:
    st.sidebar.error(f"Error connecting to database: {e}")
    st.sidebar.info("Please run the ingestion script to initialize the database: `python ingest.py`")
    db_loaded = False

# Sidebar controls & Metadata filters
st.sidebar.markdown("<h2 style='color:#a29bfe;'>⚙️ Study Filters</h2>", unsafe_allow_html=True)
st.sidebar.write("Refine search context by selecting specific textbooks metadata:")

selected_class = st.sidebar.selectbox(
    "Select Class",
    ["All Classes", "Class 6", "Class 7", "Class 8"]
)

selected_subject = st.sidebar.selectbox(
    "Select Subject",
    ["All Subjects", "History (Hindi)", "General Curriculum"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div class='glass-card' style='padding:12px; font-size:0.85rem;'>
    <h4 style='color:#a29bfe; margin-top:0;'>💡 Features Active:</h4>
    <ul>
        <li><b>Multilingual Support</b>: Ask in English/Hindi</li>
        <li><b>Hybrid Search</b>: Chroma Dense + BM25 Sparse</li>
        <li><b>Cross-Encoder Re-ranker</b>: ms-marco-MiniLM</li>
        <li><b>Semantic Cache</b>: Instant SQLite Lookup</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Build metadata filters for ChromaDB
filters = {}
if selected_class != "All Classes":
    filters["class"] = selected_class
if selected_subject != "All Subjects":
    filters["subject"] = selected_subject

# Initialize Session State Chat
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant", 
        "content": "Hello! Ask me any questions about your NCERT curriculum in English or Hindi, and I will search the textbooks to answer you.",
        "sources": [],
        "cache_info": None
    }]

# Display chat messages from history
for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        # If cache hit, show the glowing badge
        if message.get("cache_info") and message["cache_info"].get("is_cached"):
            c_info = message["cache_info"]
            st.markdown(
                f"<div class='cache-badge'>⚡ Semantic Cache Hit ({c_info['similarity']:.1%} match)</div>", 
                unsafe_allow_html=True
            )
            
        st.markdown(message["content"])
        
        # Display sources as pills
        if message["sources"]:
            st.write("")
            sources_html = "".join(f"<span class='source-pill'>📖 {s}</span>" for s in message["sources"])
            st.markdown(f"**Sources:** {sources_html}", unsafe_allow_html=True)

# Input box
if db_loaded:
    if prompt := st.chat_input("What is history? / इतिहास क्या है?"):
        # Display user message
        st.session_state.messages.append({
            "role": "user", 
            "content": prompt,
            "sources": [],
            "cache_info": None
        })
        with st.chat_message("user"):
            st.markdown(prompt)

        # Process assistant answer
        with st.chat_message("assistant"):
            with st.spinner("Searching textbooks and preparing answer..."):
                answer, sources, cache_info = process_rag_flow(
                    user_query=prompt,
                    collection=collection,
                    embedding_function=embedding_function,
                    retriever=retriever,
                    cache=cache,
                    model=model,
                    filters=filters if filters else None
                )
                
                # Show cache hit animation immediately if applicable
                if cache_info.get("is_cached"):
                    st.markdown(
                        f"<div class='cache-badge'>⚡ Semantic Cache Hit ({cache_info['similarity']:.1%} match)</div>", 
                        unsafe_allow_html=True
                    )
                
                st.markdown(answer)
                
                if sources:
                    st.write("")
                    sources_html = "".join(f"<span class='source-pill'>📖 {s}</span>" for s in sources)
                    st.markdown(f"**Sources:** {sources_html}", unsafe_allow_html=True)
                    
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "cache_info": cache_info
            })
else:
    st.warning("Application is idle. Please configure and run the ingestion process first.")