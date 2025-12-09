"""Vector database for skill storage and retrieval using ChromaDB."""

# Monkey patch pydantic.BaseSettings for older/incompatible libraries (like specific chromadb versions)
try:
    import pydantic
    try:
        from pydantic_settings import BaseSettings
    except ImportError:
        # Fallback if pydantic-settings not installed (though it should be)
        BaseSettings = None 

    if BaseSettings and not hasattr(pydantic, 'BaseSettings'):
        pydantic.BaseSettings = BaseSettings
except ImportError:
    pass

import chromadb
from typing import List, Dict, Optional
import uuid
import json
from datetime import datetime

class SkillManager:
    def __init__(self, persist_directory: str = "./chroma_db"):
        if hasattr(chromadb, 'PersistentClient'):
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            # Fallback for Chroma 0.3.x
            from chromadb.config import Settings
            self.client = chromadb.Client(Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=persist_directory
            ))

        self.collection = self.client.get_or_create_collection(
            name="skills",
            metadata={"hnsw:space": "cosine"}
        )
    
    def save_skill(self, name: str, code: str, description: str) -> str:
        """Save a new skill to the vector database."""
        skill_id = str(uuid.uuid4())
        skill_data = {
            "name": name,
            "code": code,
            "description": description
        }
        
        # Combine name and description for embedding
        text_content = f"{name}: {description}"
        
        self.collection.add(
            documents=[text_content],
            metadatas=[skill_data],
            ids=[skill_id]
        )
        return skill_id
    
    def retrieve_skill(self, query: str, n_results: int = 3) -> List[Dict]:
        """Retrieve relevant skills based on query."""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
        except Exception:
            return []
        
        skills: List[Dict] = []
        metadatas = results.get('metadatas') or []
        if metadatas and isinstance(metadatas, list):
            first = metadatas[0] if len(metadatas) > 0 else []
            if first:
                for metadata in first:
                    if isinstance(metadata, dict):
                        skills.append(metadata)
        
        return skills
    
    def list_all_skills(self) -> List[Dict]:
        """List all stored skills."""
        results = self.collection.get()
        return results['metadatas'] if results['metadatas'] else []

class EpisodicMemory:
    def __init__(self, persist_directory: str = "./chroma_db"):
        if hasattr(chromadb, 'PersistentClient'):
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            from chromadb.config import Settings
            self.client = chromadb.Client(Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=persist_directory
            ))

        self.collection = self.client.get_or_create_collection(
            name="episodic_memory",
            metadata={"hnsw:space": "cosine"}
        )

    def save_memory(self, user_id: str, content: str, metadata: Dict = None) -> str:
        """Save an interaction summary to episodic memory."""
        memory_id = str(uuid.uuid4())
        if metadata is None:
            metadata = {}
        # Trim overly long content to keep embeddings efficient
        if isinstance(content, str) and len(content) > 2000:
            content = content[:2000] + "..."
        metadata["user_id"] = user_id
        # Use ISO 8601 UTC timestamp for better sorting and readability
        metadata["timestamp"] = datetime.utcnow().isoformat() + "Z"
        # Add helpful metadata for filtering/analytics (non-breaking)
        metadata["length"] = len(content) if isinstance(content, str) else 0
        metadata.setdefault("type", "dialogue_pair")
        
        self.collection.add(
            documents=[content],
            metadatas=[metadata],
            ids=[memory_id]
        )
        return memory_id

    def retrieve_memory(self, user_id: str, query: str, n_results: int = 3) -> List[str]:
        """Retrieve relevant past interactions for a user."""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where={"user_id": user_id}
            )
            return results['documents'][0] if results['documents'] else []
        except Exception:
            return []

    def retrieve_recent_memory(self, user_id: str, n_results: int = 3) -> List[str]:
        """Retrieve the most recent past interactions for a user (fallback when no good query)."""
        try:
            # Get documents with metadata to sort by timestamp
            results = self.collection.get(where={"user_id": user_id})
            docs = results.get("documents") or []
            metas = results.get("metadatas") or []
            ids = results.get("ids") or []
            # Flatten because Chroma may return lists per id
            flat = []
            for i in range(len(ids)):
                doc = docs[i] if i < len(docs) else None
                meta = metas[i] if i < len(metas) else None
                if isinstance(doc, str) and isinstance(meta, dict):
                    ts = meta.get("timestamp", "")
                    flat.append((ts, doc))
            # Sort by timestamp desc (ISO 8601 strings sort lexicographically)
            flat.sort(key=lambda x: x[0], reverse=True)
            return [d for _, d in flat[:n_results]]
        except Exception:
            return []

class KnowledgeBase:
    """Lightweight retrieval-augmented knowledge base using ChromaDB."""
    def __init__(self, persist_directory: str = "./chroma_db"):
        if hasattr(chromadb, 'PersistentClient'):
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            from chromadb.config import Settings
            self.client = chromadb.Client(Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=persist_directory
            ))

        self.collection = self.client.get_or_create_collection(
            name="knowledge_base",
            metadata={"hnsw:space": "cosine"}
        )

    def _chunk_text(self, text: str, chunk_size: int = 800, overlap: int = 100):
        """Naive text chunking to keep embeddings efficient."""
        if not isinstance(text, str):
            return []
        text = text.strip()
        if not text:
            return []
        chunks = []
        start = 0
        n = len(text)
        while start < n:
            end = min(start + chunk_size, n)
            chunks.append(text[start:end])
            if end == n:
                break
            start = end - overlap if end - overlap > start else end
        return chunks

    def add_document(self, text: str, metadata: Optional[Dict] = None) -> int:
        """Add a free-form text document into the knowledge base. Returns number of chunks added."""
        if metadata is None:
            metadata = {}
        chunks = self._chunk_text(text)
        if not chunks:
            return 0
        ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = []
        for c in chunks:
            m = dict(metadata) if isinstance(metadata, dict) else {}
            m.setdefault("source", metadata.get("source", "user"))
            m.setdefault("timestamp", datetime.utcnow().isoformat() + "Z")
            metadatas.append(m)
        self.collection.add(documents=chunks, metadatas=metadatas, ids=ids)
        return len(chunks)

    def retrieve_knowledge(self, query: str, n_results: int = 3) -> List[Dict]:
        """Retrieve relevant knowledge chunks. Returns list of dicts with text and metadata."""
        try:
            results = self.collection.query(query_texts=[query], n_results=n_results)
            docs = results.get("documents") or []
            metas = results.get("metadatas") or []
            out: List[Dict] = []
            if docs:
                rows = zip(docs[0], metas[0] if metas else [{} for _ in docs[0]])
                for d, m in rows:
                    if isinstance(d, str):
                        out.append({"text": d, "metadata": m if isinstance(m, dict) else {}})
            return out
        except Exception:
            return []
