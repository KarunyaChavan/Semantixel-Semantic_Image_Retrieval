"""
BM25-based text search for OCR content.
Provides keyword-based search instead of semantic embedding.
"""

from rank_bm25 import BM25Okapi
import pickle
import os

class BM25TextIndex:
    """
    BM25-based full-text search index for OCR content.
    Better than semantic embeddings for keyword matching.
    """
    
    def __init__(self, index_path="db/bm25_index.pkl"):
        self.index_path = index_path
        self.bm25 = None
        self.documents = []  # Mapping: doc_id -> text
        self.doc_ids = []     # Mapping: index -> doc_id
        self.load_or_create()
    
    def load_or_create(self):
        """Load existing index or create new one"""
        if os.path.exists(self.index_path):
            try:
                with open(self.index_path, 'rb') as f:
                    data = pickle.load(f)
                    self.bm25 = data['bm25']
                    self.documents = data['documents']
                    self.doc_ids = data['doc_ids']
                print(f"✓ Loaded BM25 index with {len(self.doc_ids)} documents")
            except Exception as e:
                print(f"⚠ Error loading BM25 index: {e}. Creating new one.")
                self.bm25 = None
                self.documents = []
                self.doc_ids = []
        else:
            print("Creating new BM25 index...")
            self.bm25 = None
            self.documents = []
            self.doc_ids = []
    
    def add_document(self, doc_id, text):
        """Add or update a document in the index"""
        if not text or text.strip() == "":
            return
        
        # Tokenize: split by whitespace and convert to lowercase
        tokens = text.lower().split()
        
        # If document already exists, we'll rebuild the index
        if doc_id in self.doc_ids:
            idx = self.doc_ids.index(doc_id)
            self.documents[idx] = text
        else:
            self.documents.append(text)
            self.doc_ids.append(doc_id)
    
    def rebuild_index(self):
        """Rebuild BM25 index from documents"""
        if not self.documents:
            print("⚠ No documents to index")
            return
        
        # Tokenize all documents
        tokenized_docs = [doc.lower().split() for doc in self.documents]
        
        # Create BM25 index
        self.bm25 = BM25Okapi(tokenized_docs)
        print(f"✓ BM25 index rebuilt with {len(self.documents)} documents")
        self.save()
    
    def search(self, query, top_k=5, threshold=0.0):
        """
        Search for documents matching the query.
        
        Args:
            query (str): Search query
            top_k (int): Number of top results
            threshold (float): Minimum BM25 score (typically 0.0 for keyword search)
        
        Returns:
            list: Doc IDs sorted by relevance score
        """
        if self.bm25 is None:
            return []
        
        # Tokenize query
        tokens = query.lower().split()
        
        # Get BM25 scores
        scores = self.bm25.get_scores(tokens)
        
        # Filter by threshold and sort
        results = [
            (self.doc_ids[i], scores[i]) 
            for i in range(len(scores)) 
            if scores[i] > threshold
        ]
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k doc IDs
        return [doc_id for doc_id, score in results[:top_k]]
    
    def save(self):
        """Persist index to disk"""
        os.makedirs(os.path.dirname(self.index_path) or '.', exist_ok=True)
        with open(self.index_path, 'wb') as f:
            pickle.dump({
                'bm25': self.bm25,
                'documents': self.documents,
                'doc_ids': self.doc_ids
            }, f)
        print(f"✓ BM25 index saved to {self.index_path}")
