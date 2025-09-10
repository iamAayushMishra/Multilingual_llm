import os
import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai
from dotenv import load_dotenv

# --- CONFIGURATION ---
# Load environment variables from .env file
load_dotenv()

# Configure the Gemini API key from Streamlit secrets or .env file
try:
    # This is the preferred way for deployed Streamlit apps
    # genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    # For local development, we'll fall back to the .env file
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
except AttributeError:
    st.error("Error: GOOGLE_API_KEY not found. Please ensure you have a .env file with the key for local development.")
    st.stop()


# Same configuration as in ingest.py and query.py
CHROMA_PERSIST_DIR = "ncert_db"
COLLECTION_NAME = "ncert_books"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# --- CACHED FUNCTIONS TO LOAD MODELS ---
# Using Streamlit's caching to load models only once

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
    return collection

@st.cache_resource
def load_gemini_model():
    """Loads the Gemini Pro model."""
    model = genai.GenerativeModel('gemini-2.5-pro')
    return model

# --- BACKEND LOGIC ---
def get_rag_response(user_query, collection, model):
    """
    Finds relevant context in the database and generates a response using the LLM.
    """
    # 1. Retrieve relevant context
    results = collection.query(
        query_texts=[user_query],
        n_results=5
    )
    retrieved_documents = results['documents'][0]
    context = "\n\n".join(retrieved_documents)
    
    # 2. Prepare the prompt
    prompt_template = f"""
    You are a helpful AI assistant for students named "NCERT-Mitra".
    Your task is to answer the user's question based ONLY on the context provided below.
    If the context does not contain the answer, you MUST say "I am sorry, but I cannot find the answer to that question in the provided material."
    Do not use any external knowledge. Be concise and clear in your explanation.

    CONTEXT:
    ---
    {context}
    ---

    USER'S QUESTION:
    {user_query}

    YOUR ANSWER:
    """
    
    # 3. Generate the response
    try:
        response = model.generate_content(prompt_template)
        answer = response.text
    except Exception as e:
        answer = f"An error occurred while generating the answer: {e}"

    # 4. Get sources
    sources = set(meta['source'] for meta in results['metadatas'][0])
    
    return answer, sources

# --- STREAMLIT UI ---

# Set page configuration
st.set_page_config(page_title="NCERT-Mitra", page_icon="📚", layout="wide")

# Title of the app
st.title("📚 NCERT-Mitra AI Assistant")
st.caption("Your AI-powered study partner for NCERT books")

# Load the models and database collection
collection = load_chroma_collection()
model = load_gemini_model()

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! Ask me anything about your NCERT subjects."}]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input using the chat_input widget
if prompt := st.chat_input("What is the function of a cell?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        # Show a thinking spinner while processing
        with st.spinner("Thinking..."):
            answer, sources = get_rag_response(prompt, collection, model)
            
            # Format the sources to be displayed nicely
            source_text = "\n\n--- \n**Sources:**\n" + "\n".join(f"- *{s}*" for s in sources)
            
            full_response = answer + source_text
            st.markdown(full_response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})