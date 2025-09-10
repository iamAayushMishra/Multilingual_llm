import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm
import uuid # To generate unique IDs for each chunk

# --- CONFIGURATION ---
# 1. Directory where your PDF files are stored
PDF_SOURCE_DIR = "ncert_pdfs"

# 2. Directory where the persistent ChromaDB database will be saved
CHROMA_PERSIST_DIR = "ncert_db"

# 3. Name of the collection to be created in ChromaDB
COLLECTION_NAME = "ncert_books"

# 4. Name of the pre-trained model to use for generating embeddings
#    'all-MiniLM-L6-v2' is a great starting point: fast and good quality.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def load_documents_from_pdfs(source_dir):
    """
    Loads text content from all PDF files in the specified directory.
    
    Args:
        source_dir (str): The path to the directory containing PDF files.
        
    Returns:
        list: A list of LangChain Document objects, where each object
              represents a PDF and contains its text and metadata.
    """
    pdf_files = [f for f in os.listdir(source_dir) if f.endswith(".pdf")]
    documents = []
    print(f"Found {len(pdf_files)} PDF(s) to process.")
    
    for pdf_file in tqdm(pdf_files, desc="Loading PDFs"):
        file_path = os.path.join(source_dir, pdf_file)
        try:
            loader = PyPDFLoader(file_path)
            # load_and_split() is used here to handle PDF loading errors page by page
            pages = loader.load_and_split()
            for page in pages:
                # Add the source file name to the metadata of each page
                page.metadata['source'] = pdf_file
            documents.extend(pages)
        except Exception as e:
            print(f"Error loading {pdf_file}: {e}")
            
    return documents

def chunk_documents(documents):
    """
    Splits the loaded documents into smaller chunks for effective processing.
    
    Args:
        documents (list): A list of LangChain Document objects.
        
    Returns:
        list: A list of smaller Document chunks.
    """
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # The max number of characters in a chunk
        chunk_overlap=200, # The number of characters to overlap between chunks
        length_function=len
    )
    chunked_documents = text_splitter.split_documents(documents)
    print(f"Created {len(chunked_documents)} chunks.")
    return chunked_documents

def main():
    """
    Main function to run the data ingestion pipeline.
    """
    # --- 1. Load Documents from PDFs ---
    documents = load_documents_from_pdfs(PDF_SOURCE_DIR)
    if not documents:
        print("No documents were loaded. Please check the PDF source directory.")
        return

    # --- 2. Chunk the Documents ---
    chunked_documents = chunk_documents(documents)

    # --- 3. Initialize ChromaDB and Embedding Function ---
    print("Initializing ChromaDB and embedding model...")
    # Use Chroma's built-in SentenceTransformer embedding function
    # This will download the model automatically if you don't have it.
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME
    )

    # Initialize the persistent ChromaDB client
    db_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

    # Get or create the collection, specifying the embedding function
    collection = db_client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function
    )

    # --- 4. Add Documents to ChromaDB in Batches ---
    print("Adding document chunks to ChromaDB...")
    batch_size = 100
    total_chunks = len(chunked_documents)
    
    for i in tqdm(range(0, total_chunks, batch_size), desc="Adding to DB"):
        batch = chunked_documents[i:i + batch_size]
        
        # Prepare data for ChromaDB
        ids = [str(uuid.uuid4()) for _ in batch] # Generate a unique ID for each chunk
        texts = [doc.page_content for doc in batch]
        metadatas = [doc.metadata for doc in batch]

        # Add the batch to the collection
        collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas
        )

    print("\n--- Data Ingestion Complete! ---")
    print(f"Total documents processed: {len(documents)}")
    print(f"Total chunks created and stored: {collection.count()}")
    print(f"Database is persistently stored at: {CHROMA_PERSIST_DIR}")


if __name__ == "__main__":
    main()