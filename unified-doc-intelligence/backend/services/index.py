"""
Vector indexing service using FAISS for semantic search.
"""
import os
import logging
import pickle
from typing import List, Dict, Tuple, Optional
import numpy as np

import faiss
from langchain.schema import Document

from ..settings import settings
from .embed import embedding_service
from .storage import storage_service

logger = logging.getLogger(__name__)


class VectorIndexService:
    """
    FAISS-based vector indexing service for semantic search.
    Manages per-session indices for isolation.
    """
    
    def __init__(self):
        self.faiss_dir = settings.FAISS_DIR
        self.dimension = None  # Will be set when first vectors are added
        self._session_indices: Dict[str, faiss.Index] = {}
        self._session_metadata: Dict[str, List[Dict]] = {}
    
    def _get_index_path(self, session_id: str) -> str:
        """Get the file path for a session's FAISS index."""
        return os.path.join(self.faiss_dir, f"{session_id}.faiss")
    
    def _get_metadata_path(self, session_id: str) -> str:
        """Get the file path for a session's metadata."""
        return os.path.join(self.faiss_dir, f"{session_id}_metadata.pkl")
    
    def _load_session_index(self, session_id: str) -> bool:
        """Load a session's index from disk if it exists."""
        index_path = self._get_index_path(session_id)
        metadata_path = self._get_metadata_path(session_id)
        
        if os.path.exists(index_path) and os.path.exists(metadata_path):
            try:
                # Load FAISS index
                index = faiss.read_index(index_path)
                
                # Load metadata
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                
                self._session_indices[session_id] = index
                self._session_metadata[session_id] = metadata
                self.dimension = index.d
                
                logger.info(f"Loaded index for session {session_id}: {index.ntotal} vectors")
                return True
                
            except Exception as e:
                logger.error(f"Failed to load index for session {session_id}: {e}")
                return False
        
        return False
    
    def _save_session_index(self, session_id: str):
        """Save a session's index to disk."""
        if session_id not in self._session_indices:
            return
        
        try:
            index_path = self._get_index_path(session_id)
            metadata_path = self._get_metadata_path(session_id)
            
            # Save FAISS index
            faiss.write_index(self._session_indices[session_id], index_path)
            
            # Save metadata
            with open(metadata_path, 'wb') as f:
                pickle.dump(self._session_metadata[session_id], f)
            
            logger.info(f"Saved index for session {session_id}")
            
        except Exception as e:
            logger.error(f"Failed to save index for session {session_id}: {e}")
    
    def _create_session_index(self, session_id: str, dimension: int):
        """Create a new FAISS index for a session."""
        # Use IndexFlatIP for cosine similarity (with normalized vectors)
        index = faiss.IndexFlatIP(dimension)
        
        self._session_indices[session_id] = index
        self._session_metadata[session_id] = []
        self.dimension = dimension
        
        logger.info(f"Created new index for session {session_id} with dimension {dimension}")
    
    def add_documents(self, session_id: str, documents: List[Document]) -> bool:
        """
        Add documents to the session's vector index.
        
        Args:
            session_id: Session ID
            documents: List of LangChain Document objects with embeddings
            
        Returns:
            True if successful, False otherwise
        """
        if not documents:
            return True
        
        try:
            # Extract text content
            texts = [doc.page_content for doc in documents]
            
            # Generate embeddings
            logger.info(f"Generating embeddings for {len(texts)} documents")
            embeddings = embedding_service.embed_documents(texts)
            
            if not embeddings:
                logger.error("Failed to generate embeddings")
                return False
            
            # Convert to numpy array and normalize for cosine similarity
            vectors = np.array(embeddings, dtype=np.float32)
            faiss.normalize_L2(vectors)  # Normalize for cosine similarity
            
            # Load or create index
            if session_id not in self._session_indices:
                if not self._load_session_index(session_id):
                    self._create_session_index(session_id, vectors.shape[1])
            
            index = self._session_indices[session_id]
            
            # Check dimension compatibility
            if vectors.shape[1] != index.d:
                logger.error(f"Dimension mismatch: vectors={vectors.shape[1]}, index={index.d}")
                return False
            
            # Add vectors to index
            start_id = index.ntotal
            index.add(vectors)
            
            # Store metadata
            metadata_list = self._session_metadata[session_id]
            for i, doc in enumerate(documents):
                doc_metadata = doc.metadata.copy()
                doc_metadata.update({
                    'vector_id': start_id + i,
                    'content': doc.page_content,
                    'content_length': len(doc.page_content)
                })
                metadata_list.append(doc_metadata)
            
            # Save to disk
            self._save_session_index(session_id)
            
            logger.info(f"Added {len(documents)} documents to index for session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add documents to index: {e}")
            return False
    
    def search(self, session_id: str, query: str, top_k: int = 10) -> List[Dict]:
        """
        Perform semantic search in a session's index.
        
        Args:
            session_id: Session ID
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of search results with metadata
        """
        try:
            # Load index if not in memory
            if session_id not in self._session_indices:
                if not self._load_session_index(session_id):
                    logger.warning(f"No index found for session {session_id}")
                    return []
            
            index = self._session_indices[session_id]
            metadata_list = self._session_metadata[session_id]
            
            if index.ntotal == 0:
                return []
            
            # Generate query embedding
            query_embedding = embedding_service.embed_query(query)
            query_vector = np.array([query_embedding], dtype=np.float32)
            faiss.normalize_L2(query_vector)
            
            # Search
            k = min(top_k, index.ntotal)
            scores, indices = index.search(query_vector, k)
            
            # Format results
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx >= 0 and idx < len(metadata_list):
                    metadata = metadata_list[idx]
                    
                    result = {
                        'content': metadata.get('content', ''),
                        'score': float(score),
                        'page_number': metadata.get('page_number', 0),
                        'document': metadata.get('source', metadata.get('filename', 'Unknown')),
                        'chunk_id': metadata.get('chunk_id', f'chunk_{idx}'),
                        'metadata': metadata
                    }
                    results.append(result)
            
            logger.info(f"Found {len(results)} results for query in session {session_id}")
            return results
            
        except Exception as e:
            logger.error(f"Search failed for session {session_id}: {e}")
            return []
    
    def get_session_stats(self, session_id: str) -> Dict:
        """Get statistics for a session's index."""
        try:
            if session_id not in self._session_indices:
                if not self._load_session_index(session_id):
                    return {'vector_count': 0, 'dimension': 0, 'exists': False}
            
            index = self._session_indices[session_id]
            metadata_list = self._session_metadata[session_id]
            
            return {
                'vector_count': index.ntotal,
                'dimension': index.d,
                'metadata_count': len(metadata_list),
                'exists': True,
                'index_type': type(index).__name__
            }
            
        except Exception as e:
            logger.error(f"Failed to get stats for session {session_id}: {e}")
            return {'vector_count': 0, 'dimension': 0, 'exists': False, 'error': str(e)}
    
    def delete_session_index(self, session_id: str):
        """Delete a session's index and metadata."""
        try:
            # Remove from memory
            if session_id in self._session_indices:
                del self._session_indices[session_id]
            if session_id in self._session_metadata:
                del self._session_metadata[session_id]
            
            # Remove files
            index_path = self._get_index_path(session_id)
            metadata_path = self._get_metadata_path(session_id)
            
            for path in [index_path, metadata_path]:
                if os.path.exists(path):
                    os.remove(path)
            
            logger.info(f"Deleted index for session {session_id}")
            
        except Exception as e:
            logger.error(f"Failed to delete index for session {session_id}: {e}")
    
    def rebuild_session_index(self, session_id: str) -> bool:
        """Rebuild index for a session from stored chunks."""
        try:
            logger.info(f"Rebuilding index for session {session_id}")
            
            # Delete existing index
            self.delete_session_index(session_id)
            
            # Get chunks from storage
            chunks = storage_service.get_session_chunks(session_id)
            
            if not chunks:
                logger.warning(f"No chunks found for session {session_id}")
                return True  # Successfully rebuilt empty index
            
            # Add chunks to new index
            return self.add_documents(session_id, chunks)
            
        except Exception as e:
            logger.error(f"Failed to rebuild index for session {session_id}: {e}")
            return False


# Global vector index service
vector_index_service = VectorIndexService()


# Convenience functions
def add_documents_to_index(session_id: str, documents: List[Document]) -> bool:
    """Add documents to session's vector index."""
    return vector_index_service.add_documents(session_id, documents)


def semantic_search(session_id: str, query: str, top_k: int = 10) -> List[Dict]:
    """Perform semantic search in session."""
    return vector_index_service.search(session_id, query, top_k)


def get_index_stats(session_id: str) -> Dict:
    """Get index statistics for session."""
    return vector_index_service.get_session_stats(session_id)


def rebuild_index(session_id: str) -> bool:
    """Rebuild index for session."""
    return vector_index_service.rebuild_session_index(session_id)
