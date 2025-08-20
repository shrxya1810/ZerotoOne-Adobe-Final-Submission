"""
Strategic content selection service for better document representation.
Provides multiple strategies for selecting the most representative content from documents.
"""
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
import numpy as np

from ..settings import settings
from .storage import storage_service
from .index import semantic_search
from .embed import embedding_service

logger = logging.getLogger(__name__)


class ContentSelectionService:
    """
    Service for strategic content selection from documents.
    Provides multiple strategies to get representative content for similarity calculations.
    """
    
    def __init__(self):
        self.max_content_length = getattr(settings, 'MAX_CONTENT_SELECTION_LENGTH', 4000)
        self.target_chunk_count = getattr(settings, 'TARGET_CHUNK_COUNT', 5)
    
    def get_strategic_content(self, session_id: str, doc_id: str, doc_name: str, 
                            strategy: str = "auto") -> Tuple[str, Dict[str, Any]]:
        """
        Get strategic content representation for a document.
        
        Args:
            session_id: Session ID
            doc_id: Document ID
            doc_name: Document filename
            strategy: Selection strategy ("auto", "diverse", "semantic", "structural", "hybrid")
            
        Returns:
            Tuple of (content_text, metadata_dict)
        """
        try:
            # Get all chunks for the document
            chunks = storage_service.get_document_chunks(doc_id)
            
            if not chunks:
                logger.warning(f"No chunks found for document {doc_id}")
                return "", {"strategy": "fallback", "reason": "no_chunks"}
            
            # Choose strategy automatically if not specified
            if strategy == "auto":
                strategy = self._choose_optimal_strategy(chunks, doc_name)
            
            # Apply the selected strategy
            if strategy == "diverse":
                content, metadata = self._diverse_sampling_strategy(chunks, doc_name)
            elif strategy == "semantic":
                content, metadata = self._semantic_clustering_strategy(session_id, chunks, doc_name)
            elif strategy == "structural":
                content, metadata = self._structural_strategy(chunks, doc_name)
            elif strategy == "hybrid":
                content, metadata = self._hybrid_strategy(session_id, chunks, doc_name)
            else:
                # Default to hybrid
                content, metadata = self._hybrid_strategy(session_id, chunks, doc_name)
            
            # Add strategy info to metadata
            metadata.update({
                "strategy_used": strategy,
                "total_chunks": len(chunks),
                "content_length": len(content),
                "selection_quality": self._assess_content_quality(content, doc_name)
            })
            
            logger.info(f"📄 Strategic content for {doc_name}: {len(content)} chars using {strategy} strategy")
            
            return content, metadata
            
        except Exception as e:
            logger.error(f"Strategic content selection failed for {doc_id}: {e}")
            return "", {"strategy": "error", "error": str(e)}
    
    def _choose_optimal_strategy(self, chunks: List[Dict], doc_name: str) -> str:
        """Choose the best strategy based on document characteristics."""
        chunk_count = len(chunks)
        
        # For small documents, use all content
        if chunk_count <= 3:
            return "structural"
        
        # For medium documents, use diverse sampling
        elif chunk_count <= 10:
            return "diverse"
        
        # For large documents, use hybrid approach
        else:
            return "hybrid"
    
    def _diverse_sampling_strategy(self, chunks: List[Dict], doc_name: str) -> Tuple[str, Dict]:
        """
        Diverse sampling: Introduction + Middle + Conclusion + Key sections.
        """
        if not chunks:
            return "", {"method": "diverse", "selected_chunks": 0}
        
        # Sort chunks by page and position
        sorted_chunks = sorted(chunks, key=lambda x: (
            x.get('metadata', {}).get('page_number', 0),
            x.get('metadata', {}).get('chunk_index', 0)
        ))
        
        selected_chunks = []
        selection_info = []
        
        # 1. Introduction (first 1-2 chunks)
        intro_chunks = sorted_chunks[:2]
        selected_chunks.extend(intro_chunks)
        selection_info.extend([f"intro_{i}" for i in range(len(intro_chunks))])
        
        # 2. Conclusion (last 1-2 chunks)
        if len(sorted_chunks) > 3:
            conclusion_chunks = sorted_chunks[-2:]
            selected_chunks.extend(conclusion_chunks)
            selection_info.extend([f"conclusion_{i}" for i in range(len(conclusion_chunks))])
        
        # 3. Middle sections (evenly distributed)
        if len(sorted_chunks) > 5:
            middle_start = 2
            middle_end = len(sorted_chunks) - 2
            middle_chunks = sorted_chunks[middle_start:middle_end]
            
            # Sample evenly from middle
            sample_count = min(2, len(middle_chunks))
            if sample_count > 0:
                step = len(middle_chunks) // sample_count
                middle_selected = [middle_chunks[i * step] for i in range(sample_count)]
                selected_chunks.extend(middle_selected)
                selection_info.extend([f"middle_{i}" for i in range(len(middle_selected))])
        
        # Assemble content
        content_parts = []
        for chunk in selected_chunks:
            content_parts.append(chunk.get('content', ''))
        
        content = " ".join(content_parts)
        
        # Truncate if too long
        if len(content) > self.max_content_length:
            content = content[:self.max_content_length] + "..."
        
        metadata = {
            "method": "diverse_sampling",
            "selected_chunks": len(selected_chunks),
            "selection_info": selection_info,
            "total_available": len(chunks)
        }
        
        return content, metadata
    
    def _semantic_clustering_strategy(self, session_id: str, chunks: List[Dict], doc_name: str) -> Tuple[str, Dict]:
        """
        Semantic clustering: Find the most representative chunks using embeddings.
        """
        try:
            # Use semantic search to find most relevant chunks
            # Search for document name, abstract keywords, and topic keywords
            search_queries = [
                doc_name,
                "introduction abstract summary",
                "conclusion results findings",
                "methodology approach method"
            ]
            
            relevant_chunks = set()
            query_results = {}
            
            for query in search_queries:
                results = semantic_search(session_id, query, 3)
                query_results[query] = len(results)
                for result in results:
                    # Find matching chunk by content
                    for chunk in chunks:
                        if chunk.get('content', '') == result.get('content', ''):
                            relevant_chunks.add(id(chunk))  # Use object id as key
                            break
            
            # If no semantic matches, fall back to diverse sampling
            if not relevant_chunks:
                return self._diverse_sampling_strategy(chunks, doc_name)
            
            # Get unique chunks
            selected_chunks = []
            for chunk in chunks:
                if id(chunk) in relevant_chunks:
                    selected_chunks.append(chunk)
                if len(selected_chunks) >= self.target_chunk_count:
                    break
            
            # Assemble content
            content = " ".join([chunk.get('content', '') for chunk in selected_chunks])
            
            # Truncate if needed
            if len(content) > self.max_content_length:
                content = content[:self.max_content_length] + "..."
            
            metadata = {
                "method": "semantic_clustering",
                "selected_chunks": len(selected_chunks),
                "query_results": query_results,
                "total_available": len(chunks)
            }
            
            return content, metadata
            
        except Exception as e:
            logger.warning(f"Semantic clustering failed: {e}, falling back to diverse sampling")
            return self._diverse_sampling_strategy(chunks, doc_name)
    
    def _structural_strategy(self, chunks: List[Dict], doc_name: str) -> Tuple[str, Dict]:
        """
        Structural strategy: Focus on document structure (headings, abstracts, etc.).
        """
        # Identify chunks with structural importance
        structural_chunks = []
        regular_chunks = []
        
        for chunk in chunks:
            content = chunk.get('content', '').lower()
            
            # Check for structural indicators
            is_structural = (
                'abstract' in content[:100] or
                'introduction' in content[:100] or
                'conclusion' in content[-100:] or
                'summary' in content or
                len(content.split()) < 50  # Short chunks might be headings
            )
            
            if is_structural:
                structural_chunks.append(chunk)
            else:
                regular_chunks.append(chunk)
        
        # Select structural chunks first, then fill with regular chunks
        selected_chunks = structural_chunks[:3]
        
        if len(selected_chunks) < self.target_chunk_count:
            # Add regular chunks to reach target
            remaining_needed = self.target_chunk_count - len(selected_chunks)
            selected_chunks.extend(regular_chunks[:remaining_needed])
        
        # Assemble content
        content = " ".join([chunk.get('content', '') for chunk in selected_chunks])
        
        # Truncate if needed
        if len(content) > self.max_content_length:
            content = content[:self.max_content_length] + "..."
        
        metadata = {
            "method": "structural",
            "structural_chunks": len(structural_chunks),
            "regular_chunks": len(regular_chunks),
            "selected_chunks": len(selected_chunks),
            "total_available": len(chunks)
        }
        
        return content, metadata
    
    def _hybrid_strategy(self, session_id: str, chunks: List[Dict], doc_name: str) -> Tuple[str, Dict]:
        """
        Hybrid strategy: Combine multiple approaches for best results.
        """
        try:
            # Phase 1: Get structural chunks
            structural_content, structural_meta = self._structural_strategy(chunks, doc_name)
            
            # Phase 2: Get semantically relevant chunks
            semantic_content, semantic_meta = self._semantic_clustering_strategy(session_id, chunks, doc_name)
            
            # Phase 3: Get diverse samples
            diverse_content, diverse_meta = self._diverse_sampling_strategy(chunks, doc_name)
            
            # Combine contents intelligently
            # Priority: structural > semantic > diverse
            content_parts = []
            
            # Add abstract/introduction from structural
            if structural_content:
                content_parts.append(structural_content[:800])  # First 800 chars
            
            # Add key findings from semantic
            if semantic_content and semantic_content != structural_content:
                semantic_part = semantic_content[:1200]  # Middle 1200 chars
                if semantic_part not in structural_content:
                    content_parts.append(semantic_part)
            
            # Add diverse sample for completeness
            if diverse_content:
                diverse_part = diverse_content[:1000]  # Last 1000 chars
                if diverse_part not in " ".join(content_parts):
                    content_parts.append(diverse_part)
            
            # Combine and clean up
            combined_content = " ".join(content_parts)
            
            # Remove duplicates and truncate
            combined_content = self._remove_duplicate_sentences(combined_content)
            
            if len(combined_content) > self.max_content_length:
                combined_content = combined_content[:self.max_content_length] + "..."
            
            metadata = {
                "method": "hybrid",
                "structural_meta": structural_meta,
                "semantic_meta": semantic_meta,
                "diverse_meta": diverse_meta,
                "final_length": len(combined_content),
                "total_available": len(chunks)
            }
            
            return combined_content, metadata
            
        except Exception as e:
            logger.warning(f"Hybrid strategy failed: {e}, falling back to diverse sampling")
            return self._diverse_sampling_strategy(chunks, doc_name)
    
    def _remove_duplicate_sentences(self, text: str) -> str:
        """Remove duplicate sentences to avoid redundancy."""
        sentences = re.split(r'[.!?]+', text)
        unique_sentences = []
        seen = set()
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:  # Ignore very short fragments
                sentence_key = sentence.lower()[:50]  # Use first 50 chars as key
                if sentence_key not in seen:
                    seen.add(sentence_key)
                    unique_sentences.append(sentence)
        
        return ". ".join(unique_sentences) + "."
    
    def _assess_content_quality(self, content: str, doc_name: str) -> float:
        """Assess the quality of selected content (0.0 to 1.0)."""
        if not content:
            return 0.0
        
        quality_score = 0.0
        
        # Length quality (not too short, not too long)
        length_quality = min(1.0, len(content) / 2000)  # Optimal around 2000 chars
        quality_score += length_quality * 0.3
        
        # Diversity quality (variety of words)
        words = content.split()
        unique_words = len(set(word.lower() for word in words))
        diversity_quality = min(1.0, unique_words / max(len(words), 1))
        quality_score += diversity_quality * 0.3
        
        # Structure quality (has introduction/conclusion indicators)
        structure_indicators = ['introduction', 'abstract', 'conclusion', 'summary', 'findings']
        structure_quality = sum(1 for indicator in structure_indicators if indicator in content.lower()) / len(structure_indicators)
        quality_score += structure_quality * 0.4
        
        return min(1.0, quality_score)


# Global service instance
content_selector = ContentSelectionService()


# Convenience function
def get_strategic_document_content(session_id: str, doc_id: str, doc_name: str, 
                                 strategy: str = "auto") -> Tuple[str, Dict[str, Any]]:
    """
    Get strategic content for a document using the global service.
    
    Args:
        session_id: Session ID
        doc_id: Document ID
        doc_name: Document filename
        strategy: Selection strategy
        
    Returns:
        Tuple of (content_text, metadata_dict)
    """
    return content_selector.get_strategic_content(session_id, doc_id, doc_name, strategy)