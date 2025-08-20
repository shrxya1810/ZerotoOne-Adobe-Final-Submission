"""
Embedding service with MPNet as default and global caching.
"""
import os
import hashlib
import pickle
from typing import List, Optional, Dict, Any
from functools import lru_cache
import numpy as np
import logging

from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from ..settings import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Singleton embedding service with model caching and embedding caching.
    Supports MPNet (default) and MiniLM fallback.
    """
    
    _instance = None
    _embedding_model = None
    _model_name = None
    _embedding_cache: Dict[str, np.ndarray] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingService, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._embedding_model is None:
            self._load_model()
    
    def _load_model(self):
        """Load the embedding model based on settings."""
        model_name = settings.EMBEDDING_MODEL
        
        try:
            logger.info(f"Loading embedding model: {model_name}")
            
            # Set cache folder to avoid version conflicts
            cache_folder = os.path.join(settings.TEMP_DIR, "model_cache")
            os.makedirs(cache_folder, exist_ok=True)
            
            if "mpnet" in model_name.lower():
                # MPNet model (default)
                self._embedding_model = SentenceTransformerEmbeddings(
                    model_name="multi-qa-mpnet-base-dot-v1",
                    cache_folder=cache_folder
                )
                self._model_name = "multi-qa-mpnet-base-dot-v1"
                logger.info("✅ MPNet model loaded successfully")
                
            elif "minilm" in model_name.lower():
                # MiniLM fallback
                self._embedding_model = SentenceTransformerEmbeddings(
                    model_name="all-MiniLM-L6-v2",
                    cache_folder=cache_folder
                )
                self._model_name = "all-MiniLM-L6-v2"
                logger.info("✅ MiniLM model loaded successfully")
                
            else:
                # Try the exact model name from settings
                self._embedding_model = SentenceTransformerEmbeddings(
                    model_name=model_name,
                    cache_folder=cache_folder
                )
                self._model_name = model_name
                logger.info(f"✅ Custom model {model_name} loaded successfully")
                
        except Exception as e:
            logger.warning(f"Failed to load {model_name}: {e}")
            
            # Try to clear cache and reload
            try:
                import shutil
                cache_folder = os.path.join(settings.TEMP_DIR, "model_cache")
                if os.path.exists(cache_folder):
                    shutil.rmtree(cache_folder)
                    logger.info("Cleared model cache, retrying...")
                    os.makedirs(cache_folder, exist_ok=True)
                
                # Fallback to MiniLM with fresh cache
                self._embedding_model = SentenceTransformerEmbeddings(
                    model_name="all-MiniLM-L6-v2",
                    cache_folder=cache_folder
                )
                self._model_name = "all-MiniLM-L6-v2"
                logger.info("✅ Fallback MiniLM model loaded successfully")
                
            except Exception as fallback_error:
                logger.error(f"Failed to load fallback model: {fallback_error}")
                raise RuntimeError("Could not load any embedding model")
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.md5(f"{self._model_name}:{text}".encode()).hexdigest()
    
    def _manage_cache_size(self):
        """Remove oldest entries if cache exceeds size limit."""
        if len(self._embedding_cache) > settings.EMBEDDING_CACHE_SIZE:
            # Remove 20% of oldest entries (simple FIFO)
            remove_count = len(self._embedding_cache) // 5
            keys_to_remove = list(self._embedding_cache.keys())[:remove_count]
            for key in keys_to_remove:
                del self._embedding_cache[key]
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents with caching.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        # Check cache first
        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text)
            if cache_key in self._embedding_cache:
                embeddings.append(self._embedding_cache[cache_key].tolist())
            else:
                embeddings.append(None)  # Placeholder
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Compute embeddings for uncached texts
        if uncached_texts:
            try:
                new_embeddings = self._embedding_model.embed_documents(uncached_texts)
                
                # Store in cache and update results
                for idx, embedding in enumerate(new_embeddings):
                    text_idx = uncached_indices[idx]
                    text = uncached_texts[idx]
                    cache_key = self._get_cache_key(text)
                    
                    # Store as numpy array for efficiency
                    embedding_array = np.array(embedding)
                    self._embedding_cache[cache_key] = embedding_array
                    embeddings[text_idx] = embedding
                
                # Manage cache size
                self._manage_cache_size()
                
            except Exception as e:
                logger.error(f"Error computing embeddings: {e}")
                raise
        
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text with caching.
        
        Args:
            text: Query text to embed
            
        Returns:
            Embedding vector
        """
        cache_key = self._get_cache_key(text)
        
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key].tolist()
        
        try:
            embedding = self._embedding_model.embed_query(text)
            
            # Store in cache
            embedding_array = np.array(embedding)
            self._embedding_cache[cache_key] = embedding_array
            
            # Manage cache size
            self._manage_cache_size()
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error computing query embedding: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        return {
            "model_name": self._model_name,
            "cache_size": len(self._embedding_cache),
            "max_cache_size": settings.EMBEDDING_CACHE_SIZE,
            "cache_hit_ratio": self._calculate_cache_hit_ratio()
        }
    
    def _calculate_cache_hit_ratio(self) -> float:
        """Calculate cache hit ratio (simplified)."""
        # This is a placeholder - in production you'd track hits/misses
        return min(len(self._embedding_cache) / max(settings.EMBEDDING_CACHE_SIZE, 1), 1.0)
    
    def clear_cache(self):
        """Clear the embedding cache."""
        self._embedding_cache.clear()
        logger.info("Embedding cache cleared")


class ChunkingService:
    """
    Document chunking service optimized for the embedding model.
    """
    
    def __init__(self):
        # Optimize chunk size based on model
        embedding_service = EmbeddingService()
        model_name = embedding_service._model_name
        
        if "mpnet" in model_name.lower():
            # MPNet works well with larger chunks
            self.chunk_size = settings.CHUNK_SIZE
            self.chunk_overlap = settings.CHUNK_OVERLAP
        else:
            # MiniLM works better with smaller chunks
            self.chunk_size = min(settings.CHUNK_SIZE, 512)
            self.chunk_overlap = min(settings.CHUNK_OVERLAP, 64)
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ";", " ", ""]
        )
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Chunk documents into smaller pieces.
        
        Args:
            documents: List of LangChain Document objects
            
        Returns:
            List of chunked Document objects
        """
        try:
            chunks = self.text_splitter.split_documents(documents)
            
            # Add chunk metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    "chunk_id": f"chunk_{i}",
                    "chunk_size": len(chunk.page_content),
                    "chunk_index": i
                })
            
            logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
            return chunks
            
        except Exception as e:
            logger.error(f"Error chunking documents: {e}")
            raise
    
    def chunk_text(self, text: str, metadata: Optional[Dict] = None) -> List[Document]:
        """
        Chunk a single text string.
        
        Args:
            text: Text to chunk
            metadata: Optional metadata to add to chunks
            
        Returns:
            List of Document objects
        """
        if metadata is None:
            metadata = {}
        
        try:
            texts = self.text_splitter.split_text(text)
            
            documents = []
            for i, chunk_text in enumerate(texts):
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    "chunk_id": f"chunk_{i}",
                    "chunk_size": len(chunk_text),
                    "chunk_index": i
                })
                
                documents.append(Document(
                    page_content=chunk_text,
                    metadata=chunk_metadata
                ))
            
            return documents
            
        except Exception as e:
            logger.error(f"Error chunking text: {e}")
            raise


# Global instances (singletons)
embedding_service = EmbeddingService()
chunking_service = ChunkingService()


# Convenience functions
def embed_documents(texts: List[str]) -> List[List[float]]:
    """Embed documents using the global embedding service."""
    return embedding_service.embed_documents(texts)


def embed_query(text: str) -> List[float]:
    """Embed a query using the global embedding service."""
    return embedding_service.embed_query(text)


def chunk_documents(documents: List[Document]) -> List[Document]:
    """Chunk documents using the global chunking service."""
    return chunking_service.chunk_documents(documents)


def chunk_text(text: str, metadata: Optional[Dict] = None) -> List[Document]:
    """Chunk text using the global chunking service."""
    return chunking_service.chunk_text(text, metadata)


def get_embedding_info() -> Dict[str, Any]:
    """Get information about the embedding service."""
    return embedding_service.get_model_info()
