import os
import math
import re
from typing import List, Dict, Any
from src.compliance.config import CHROMA_DB_DIR
from src.compliance.rules_data import ECOA_REGULATION_B_RULES

# Try to import chromadb. If it fails, we fall back to a robust custom pure-Python vector search.
CHROMA_AVAILABLE = False
try:
    import chromadb
    CHROMA_AVAILABLE = True
except Exception as e:
    print(f"[Compliance Auditor] chromadb import/initialization failed ({e}). Falling back to built-in pure-Python TF-IDF matcher.")

class PurePythonVectorDB:
    """
    A pure-Python TF-IDF vector database fallback.
    Ensures that if chromadb fails to install or run, the RAG system still works flawlessly.
    """
    def __init__(self):
        self.documents = []
        self.idf = {}
        self.vocab = set()
        
    def _tokenize(self, text: str) -> List[str]:
        # Lowercase and split on non-alphanumeric characters
        return re.findall(r'\b\w+\b', text.lower())

    def add_documents(self, docs: List[Dict[str, Any]]):
        self.documents = docs
        # Compute IDF
        doc_count = len(docs)
        df = {}
        
        # Build vocabulary and document frequency
        for doc in docs:
            tokens = set(self._tokenize(doc["content"] + " " + doc["title"]))
            for token in tokens:
                df[token] = df.get(token, 0) + 1
                self.vocab.add(token)
                
        # Calculate IDF
        for token, count in df.items():
            self.idf[token] = math.log((1 + doc_count) / (1 + count)) + 1

    def _get_tf_idf_vector(self, text: str) -> Dict[str, float]:
        tokens = self._tokenize(text)
        tf = {}
        for token in tokens:
            if token in self.vocab:
                tf[token] = tf.get(token, 0) + 1
                
        vector = {}
        for token, count in tf.items():
            # TF-IDF = (count/len) * IDF
            vector[token] = (count / len(tokens)) * self.idf[token]
        return vector

    def _cosine_similarity(self, vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
        intersection = set(vec1.keys()) & set(vec2.keys())
        numerator = sum([vec1[x] * vec2[x] for x in intersection])
        
        sum1 = sum([val ** 2 for val in vec1.values()])
        sum2 = sum([val ** 2 for val in vec2.values()])
        
        denominator = math.sqrt(sum1) * math.sqrt(sum2)
        if not denominator:
            return 0.0
        return numerator / denominator

    def search(self, query: str, limit: int = 2) -> List[Dict[str, Any]]:
        query_vec = self._get_tf_idf_vector(query)
        if not query_vec:
            # Fallback to simple keyword overlap if tf-idf vector is empty
            return self.documents[:limit]
            
        scored_docs = []
        for doc in self.documents:
            doc_text = doc["title"] + " " + doc["content"]
            doc_vec = self._get_tf_idf_vector(doc_text)
            score = self._cosine_similarity(query_vec, doc_vec)
            scored_docs.append((score, doc))
            
        # Sort by score descending
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scored_docs[:limit]]


class RegulatoryVectorDB:
    def __init__(self):
        self.chroma_client = None
        self.collection = None
        self.fallback_db = None
        
        if CHROMA_AVAILABLE:
            try:
                # Ensure the data directory exists
                os.makedirs(CHROMA_DB_DIR, exist_ok=True)
                self.chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
                self.collection = self.chroma_client.get_or_create_collection("ecoa_rules")
                print("[Compliance Auditor] Initialized ChromaDB at", CHROMA_DB_DIR)
            except Exception as e:
                print(f"[Compliance Auditor] Failed to start ChromaDB: {e}. Using pure-Python fallback.")
                self.chroma_client = None
                
        if self.chroma_client is None:
            self.fallback_db = PurePythonVectorDB()
            print("[Compliance Auditor] Initialized pure-Python Vector DB fallback.")

    def seed_database(self) -> int:
        """
        Seeds the vector store with Regulation B ECOA guidelines.
        Returns the number of documents added.
        """
        if self.collection:
            try:
                # Clear existing items to prevent duplicates
                existing = self.collection.get()
                if existing and existing["ids"]:
                    self.collection.delete(ids=existing["ids"])
                    
                ids = [rule["id"] for rule in ECOA_REGULATION_B_RULES]
                documents = [rule["content"] for rule in ECOA_REGULATION_B_RULES]
                metadatas = [{"title": rule["title"], "category": rule["category"]} for rule in ECOA_REGULATION_B_RULES]
                
                # Chroma uses default MiniLM embeddings when embedding_function=None
                self.collection.add(
                    ids=ids,
                    documents=documents,
                    metadatas=metadatas
                )
                print(f"[Compliance Auditor] Successfully seeded {len(ids)} rules into ChromaDB.")
                return len(ids)
            except Exception as e:
                print(f"[Compliance Auditor] Failed seeding to ChromaDB: {e}. Seeding to fallback DB.")
                
        # Seed fallback database
        self.fallback_db = PurePythonVectorDB()
        self.fallback_db.add_documents(ECOA_REGULATION_B_RULES)
        print(f"[Compliance Auditor] Successfully seeded {len(ECOA_REGULATION_B_RULES)} rules into fallback vector store.")
        return len(ECOA_REGULATION_B_RULES)

    def search_rules(self, query: str, limit: int = 2) -> List[Dict[str, Any]]:
        """
        Queries the vector store for rules relevant to the input query.
        """
        if self.collection:
            try:
                results = self.collection.query(
                    query_texts=[query],
                    n_results=limit
                )
                
                retrieved_rules = []
                if results and results["documents"] and results["documents"][0]:
                    for i in range(len(results["ids"][0])):
                        retrieved_rules.append({
                            "id": results["ids"][0][i],
                            "title": results["metadatas"][0][i]["title"],
                            "content": results["documents"][0][i],
                            "category": results["metadatas"][0][i]["category"]
                        })
                    return retrieved_rules
            except Exception as e:
                print(f"[Compliance Auditor] ChromaDB search error: {e}. Searching fallback.")
                
        # Fallback search
        if not self.fallback_db or not self.fallback_db.documents:
            # Seed on the fly if not seeded
            self.seed_database()
            
        return self.fallback_db.search(query, limit=limit)

# Global singleton instance
vector_db = RegulatoryVectorDB()
