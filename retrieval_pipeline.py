import os
import math
import numpy as np
from collections import Counter
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv

load_dotenv()
DEFAULT_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "4"))
DEFAULT_DENSE_K = int(os.getenv("RETRIEVAL_DENSE_K", "15"))
DEFAULT_SPARSE_K = int(os.getenv("RETRIEVAL_SPARSE_K", "15"))

class SimpleBM25:
    def __init__(self, documents, metadatas, ids):
        """
        Simple BM25 implementation for self-contained sparse search.
        
        Args:
            documents (list of str): The text contents of all chunks.
            metadatas (list of dict): The metadata associated with each chunk.
            ids (list of str): ChromaDB document IDs.
        """
        self.documents = documents
        self.metadatas = metadatas
        self.ids = ids
        self.corpus_size = len(documents)
        
        # Tokenize corpus: lowercase and split by non-alphanumeric characters
        self.tokenized_corpus = [self._tokenize(doc) for doc in documents]
        self.doc_lengths = [len(doc) for doc in self.tokenized_corpus]
        self.avg_doc_len = sum(self.doc_lengths) / self.corpus_size if self.corpus_size > 0 else 0
        
        self.k1 = 1.5
        self.b = 0.75
        
        self.doc_freqs = []
        self.idf = {}
        self._initialize()

    def _tokenize(self, text):
        # Basic alphanumeric word tokenizer
        words = []
        current_word = []
        for char in text.lower():
            if char.isalnum():
                current_word.append(char)
            else:
                if current_word:
                    words.append("".join(current_word))
                    current_word = []
        if current_word:
            words.append("".join(current_word))
        return words

    def _initialize(self):
        nd = {}
        for doc in self.tokenized_corpus:
            frequencies = Counter(doc)
            self.doc_freqs.append(frequencies)
            for word in frequencies:
                nd[word] = nd.get(word, 0) + 1
                
        for word, freq in nd.items():
            # Adjusted BM25 IDF formulation
            self.idf[word] = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)

    def search(self, query, n_results=15):
        tokenized_query = self._tokenize(query)
        scores = [0.0] * self.corpus_size
        
        for word in tokenized_query:
            if word not in self.idf:
                continue
            idf_val = self.idf[word]
            for i, freqs in enumerate(self.doc_freqs):
                tf = freqs.get(word, 0)
                denom = tf + self.k1 * (1.0 - self.b + self.b * self.doc_lengths[i] / self.avg_doc_len)
                scores[i] += idf_val * (tf * (self.k1 + 1.0)) / denom
                
        # Sort and return top results
        ranked_indices = np.argsort(scores)[::-1][:n_results]
        
        results = []
        for idx in ranked_indices:
            if scores[idx] > 0.0:  # Only return documents with some match
                results.append({
                    "document": self.documents[idx],
                    "metadata": self.metadatas[idx],
                    "id": self.ids[idx],
                    "score": scores[idx]
                })
        return results


class AdvancedRetrievalPipeline:
    def __init__(self, chroma_collection, cross_encoder_model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Advanced retrieval pipeline combining dense Chroma search, sparse BM25, and Cross-Encoder re-ranking.
        """
        self.collection = chroma_collection
        print(f"Loading Cross-Encoder model: {cross_encoder_model_name}...")
        self.cross_encoder = CrossEncoder(cross_encoder_model_name)
        
        # Load all documents to initialize BM25
        print("Fetching documents from ChromaDB to build BM25 index...")
        db_data = self.collection.get()
        self.bm25 = None
        
        if db_data and db_data['documents']:
            self.bm25 = SimpleBM25(
                documents=db_data['documents'],
                metadatas=db_data['metadatas'],
                ids=db_data['ids']
            )
            print(f"BM25 index initialized with {len(db_data['documents'])} chunks.")
        else:
            print("Warning: ChromaDB collection is empty. BM25 not initialized.")

    def refresh_bm25(self):
        """Re-fetches database documents to refresh the BM25 index (e.g. after new ingestion)."""
        db_data = self.collection.get()
        if db_data and db_data['documents']:
            self.bm25 = SimpleBM25(
                documents=db_data['documents'],
                metadatas=db_data['metadatas'],
                ids=db_data['ids']
            )
            print(f"BM25 index refreshed with {len(db_data['documents'])} chunks.")

    def retrieve_hybrid_and_rerank(self, query, top_k=None, dense_k=None, sparse_k=None, rrf_k=60, filters=None):
        """
        Retrieve context chunks using hybrid (dense + sparse) search, fuse with RRF, and re-rank with Cross-Encoder.
        """
        if top_k is None:
            top_k = DEFAULT_TOP_K
        if dense_k is None:
            dense_k = DEFAULT_DENSE_K
        if sparse_k is None:
            sparse_k = DEFAULT_SPARSE_K

        # --- 1. Dense (Vector) Search ---
        # If metadata filters are provided, pass them to collection query
        dense_results = self.collection.query(
            query_texts=[query],
            n_results=dense_k,
            where=filters
        )
        
        dense_docs = []
        if dense_results and dense_results['documents'] and dense_results['documents'][0]:
            for i in range(len(dense_results['documents'][0])):
                dense_docs.append({
                    "document": dense_results['documents'][0][i],
                    "metadata": dense_results['metadatas'][0][i],
                    "id": dense_results['ids'][0][i]
                })

        # --- 2. Sparse (BM25) Search ---
        sparse_docs = []
        if self.bm25:
            # Note: Filters can be manually applied to BM25 results
            all_sparse = self.bm25.search(query, n_results=sparse_k * 3) # search more to filter
            if filters:
                # Apply metadata filters
                filtered_sparse = []
                for doc in all_sparse:
                    match = True
                    for key, val in filters.items():
                        if doc['metadata'].get(key) != val:
                            match = False
                            break
                    if match:
                        filtered_sparse.append(doc)
                sparse_docs = filtered_sparse[:sparse_k]
            else:
                sparse_docs = all_sparse[:sparse_k]

        # --- 3. Reciprocal Rank Fusion (RRF) ---
        rrf_scores = {}
        doc_details = {}  # Map ID -> doc details
        
        # Dense ranking
        for rank, doc in enumerate(dense_docs):
            doc_id = doc['id']
            doc_details[doc_id] = doc
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (rrf_k + (rank + 1))
            
        # Sparse ranking
        for rank, doc in enumerate(sparse_docs):
            doc_id = doc['id']
            doc_details[doc_id] = doc
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (rrf_k + (rank + 1))

        # Sort by RRF score and get top candidates for re-ranking
        fused_candidates_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:dense_k + sparse_k]
        fused_candidates = [doc_details[doc_id] for doc_id in fused_candidates_ids]

        if not fused_candidates:
            return [], []

        # --- 4. Cross-Encoder Re-ranking ---
        pairs = [(query, doc['document']) for doc in fused_candidates]
        scores = self.cross_encoder.predict(pairs)
        
        # Add score to candidates
        for idx, score in enumerate(scores):
            fused_candidates[idx]['cross_score'] = float(score)
            
        # Sort candidates by Cross-Encoder score
        reranked_candidates = sorted(fused_candidates, key=lambda x: x['cross_score'], reverse=True)
        
        # Select top_k results
        final_results = reranked_candidates[:top_k]
        
        # Extract documents and sources
        retrieved_documents = [doc['document'] for doc in final_results]
        sources = list(set(doc['metadata']['source'] for doc in final_results))
        
        return retrieved_documents, sources
