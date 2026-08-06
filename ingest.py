import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm
import uuid 
from dotenv import load_dotenv

# Load environment configuration variables
load_dotenv()

PDF_SOURCE_DIR = os.getenv("PDF_SOURCE_DIR", "textbook_pdfs")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "ncert_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "ncert_books")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "paraphrase-multilingual-MiniLM-L12-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))


def extract_metadata(pdf_name):
    """
    Extracts class, subject, and chapter information from NCERT PDF filenames.
    e.g. 'ihga101.pdf' -> class: 6, subject: History (Hindi), chapter: Chapter 1
    """
    name = pdf_name.lower().replace(".pdf", "")
    metadata = {
        "source": pdf_name,
        "class": "Unknown Class",
        "subject": "Unknown Subject",
        "chapter": "Unknown Chapter"
    }
    
    # Check code prefixes
    # Standard NCERT prefixes:
    # ihga1 -> Class 6 History (Hindi: Hamare Atit - I)
    # ihga2 -> Class 7 History (Hindi: Hamare Atit - II)
    # ihga3 -> Class 8 History (Hindi: Hamare Atit - III)
    if len(name) >= 5:
        book_code = name[:-2]  # e.g., 'ihga1'
        chapter_code = name[-2:] # e.g., '01'
        
        # Match class and subject based on prefix
        if book_code.startswith("ihga"):
            # History in Hindi (Hamare Atit)
            metadata["subject"] = "History (Hindi)"
            class_num = book_code.replace("ihga", "")
            if class_num == "1":
                metadata["class"] = "Class 6"
            elif class_num == "2":
                metadata["class"] = "Class 7"
            elif class_num == "3":
                metadata["class"] = "Class 8"
            else:
                metadata["class"] = f"Class {class_num}" if class_num.isdigit() else "Class 6"
                
            try:
                metadata["chapter"] = f"Chapter {int(chapter_code)}"
            except ValueError:
                metadata["chapter"] = f"Chapter {chapter_code}"
        else:
            # General fallback if code matches other standard NCERT prefix formats
            metadata["class"] = "Class 6"
            metadata["subject"] = "General Curriculum"
            metadata["chapter"] = name
    else:
        metadata["class"] = "Class 6"
        metadata["subject"] = "General Curriculum"
        metadata["chapter"] = name
        
    return metadata


def load_documents_from_pdfs(source_dir):
    """
    Loads text content from all PDF files in the specified directory.
    
    Args:
        source_dir (str): The path to the directory containing PDF files.
        
    Returns:
        list: A list of LangChain Document objects, where each object
              represents a PDF and contains its text and metadata.
    """
    if not os.path.exists(source_dir):
        print(f"Error: Directory '{source_dir}' does not exist.")
        return []
        
    pdf_files = [f for f in os.listdir(source_dir) if f.endswith(".pdf")]
    documents = []
    print(f"Found {len(pdf_files)} PDF(s) to process in '{source_dir}'.")
    
    for pdf_file in tqdm(pdf_files, desc="Loading PDFs"):
        file_path = os.path.join(source_dir, pdf_file)
        try:
            loader = PyPDFLoader(file_path)
            pages = loader.load_and_split()
            
            # Extract and inject rich metadata for each page
            file_meta = extract_metadata(pdf_file)
            for page in pages:
                page.metadata.update(file_meta)
                
            documents.extend(pages)
        except Exception as e:
            print(f"Error loading {pdf_file}: {e}")
            
    return documents

def chunk_documents(documents):
    """
    Splits the loaded documents into smaller chunks for effective processing.
    Includes custom separators to handle Hindi full stop (purna viram '।').
    
    Args:
        documents (list): A list of LangChain Document objects.
        
    Returns:
        list: A list of smaller Document chunks.
    """
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        # Handing both Hindi purna viram (।), English period (.), and paragraph spacing
        separators=["\n\n", "\n", "। ", "।", ". ", ".", " ", ""]
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
    # Use Chroma's built-in SentenceTransformer embedding function with multilingual model
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
        ids = [str(uuid.uuid4()) for _ in batch]
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