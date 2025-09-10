# 📚 NCERT-Mitra: An AI Learning Assistant

NCERT-Mitra is an AI-powered chatbot designed to help students in classes 6-8 understand their NCERT curriculum. Using a Retrieval-Augmented Generation (RAG) approach, it answers questions based on the actual content of the textbooks, providing accurate and context-aware responses with source references.

![NCERT-Mitra Screenshot](https://i.imgur.com/39hD12T.png)

## ✨ Features

-   **Interactive Chat UI**: A simple and intuitive web interface built with Streamlit.
-   **Context-Aware Answers**: Utilizes a RAG pipeline to ensure answers are grounded in the provided NCERT textbook content.
-   **Source Referencing**: Each answer is accompanied by the source PDF file(s) from which the information was retrieved.
-   **Easy Data Ingestion**: A simple script (`ingest.py`) to process your PDF textbooks and build the knowledge base.
-   **Modular and Extendable**: Built with modern tools that are easy to understand and build upon.

## 🛠️ Tech Stack

-   **Backend**: Python
-   **Web UI**: Streamlit
-   **LLM**: Google Gemini Pro
-   **Vector Database**: ChromaDB
-   **Embedding Model**: `all-MiniLM-L6-v2` (via Sentence Transformers)
-   **PDF Processing**: LangChain, PyPDF

## 🏛️ Project Architecture

The application uses a Retrieval-Augmented Generation (RAG) architecture:

1.  **Data Ingestion (Offline)**: All NCERT PDFs are processed, split into manageable chunks, converted into vector embeddings, and stored in a ChromaDB vector database.
2.  **User Query (Online)**:
    -   A user asks a question in the Streamlit UI.
    -   The question is converted into a vector embedding.
    -   ChromaDB is searched to find the most relevant text chunks from the textbooks.
    -   These chunks (the "context") and the user's original question are passed to the Gemini LLM in a carefully crafted prompt.
    -   Gemini generates an answer based *only* on the provided context.
    -   The final answer and its sources are displayed to the user.

## 🚀 Getting Started

Follow these instructions to set up and run the project on your local machine.

### 1. Prerequisites

-   [Git](https://git-scm.com/downloads)
-   [Python 3.9+](https://www.python.org/downloads/)

### 2. Installation & Setup

**1. Clone the repository:**
```bash
git clone [https://github.com/your-username/ncert-mitra.git](https://github.com/your-username/ncert-mitra.git)
cd ncert-mitra
```

### 2.venv
```bash
python -m venv .venv
.\.venv\Scripts\activate
```

### 3. Install dependencies:
```
streamlit
chromadb
sentence-transformers
langchain
langchain-community
pypdf
tqdm
google-generativeai
python-dotenv
```

### 4.env
```
GOOGLE_API_KEY="YOUR_API_KEY_HERE"
```

### 5.run
```
python ingest.py
streamlit run app.py
```
