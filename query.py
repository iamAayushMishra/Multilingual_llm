import os
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai
from dotenv import load_dotenv

# --- CONFIGURATION ---
# Load environment variables from .env file
load_dotenv()

# Configure the Gemini API key
try:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
except AttributeError:
    print("Error: GOOGLE_API_KEY not found. Please ensure you have a .env file with the key.")
    exit()


# Same configuration as in ingest.py
CHROMA_PERSIST_DIR = "ncert_db"
COLLECTION_NAME = "ncert_books"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# --- MAIN APPLICATION LOGIC ---

def main():
    """
    Main function to run the query application.
    """
    # --- 1. Initialize ChromaDB Client and Embedding Function ---
    print("Initializing ChromaDB client...")
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME
    )

    db_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

    # Get the existing collection
    collection = db_client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function
    )
    
    # --- 2. Initialize Gemini Model ---
    print("Initializing Gemini Pro model...")
    model = genai.GenerativeModel('gemini-2.5-pro')
    
    print("\n--- NCERT-Mitra AI Assistant ---")
    print("Ask any question about your NCERT books.")
    print("Type 'exit' to quit.\n")

    # --- 3. Interactive Q&A Loop ---
    while True:
        user_query = input("You: ")
        if user_query.lower() == 'exit':
            print("Thank you for using NCERT-Mitra. Goodbye!")
            break
        
        if not user_query.strip():
            continue

        # --- 4. Retrieve Relevant Context from ChromaDB ---
        print("Searching for relevant context...")
        # Query the collection to get the 5 most relevant chunks
        results = collection.query(
            query_texts=[user_query],
            n_results=5
        )
        
        retrieved_documents = results['documents'][0]
        context = "\n\n".join(retrieved_documents)
        
        # --- 5. Generate the Prompt for the LLM ---
        # This is a crucial step known as Prompt Engineering
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

        # --- 6. Call the LLM to Generate an Answer ---
        print("Generating answer...")
        try:
            response = model.generate_content(prompt_template)
            answer = response.text
        except Exception as e:
            answer = f"An error occurred while generating the answer: {e}"

        # --- 7. Display the Answer and Sources ---
        print("\nNCERT-Mitra:")
        print(answer)
        
        # Display the sources of the retrieved documents
        print("\n--- Sources ---")
        sources = set(meta['source'] for meta in results['metadatas'][0])
        for source in sources:
            print(f"- {source}")
        print("\n" + "="*50 + "\n")


if __name__ == "__main__":
    main()