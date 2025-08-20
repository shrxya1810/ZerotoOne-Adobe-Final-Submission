"""
Knowledge graph router for generating and visualizing document relationships.
"""
from fastapi import APIRouter, HTTPException, Query
import logging
import re
import numpy as np
from typing import List, Dict, Set, Tuple
from collections import defaultdict
import networkx as nx
try:
    import community as community_louvain
except ImportError:
    community_louvain = None
import math
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
import time
from functools import lru_cache

from ..settings import settings
from ..services.storage import storage_service
from ..services.embed import embedding_service
from ..services.index import semantic_search
from ..models.schemas import KnowledgeGraphResponse, GraphNode, GraphEdge

logger = logging.getLogger(__name__)

router = APIRouter()


@lru_cache(maxsize=1000)
def preprocess_text_fast(text: str) -> str:
    """Fast cached text preprocessing without NLTK dependency."""
    try:
        # Convert to lowercase and remove special characters in one pass
        text = re.sub(r'[^a-zA-Z\s]', ' ', text.lower())
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Fast stopword removal using simple set
        basic_stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'
        }
        words = [word for word in text.split() if word not in basic_stopwords and len(word) > 2]
        
        return ' '.join(words)
    except Exception as e:
        logger.warning(f"Fast text preprocessing failed: {e}")
        return text


def preprocess_text(text: str) -> str:
    """Legacy function - redirects to fast version."""
    return preprocess_text_fast(text)


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    try:
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 > 0 and norm2 > 0:
            return dot_product / (norm1 * norm2)
        return 0.0
    except Exception as e:
        logger.warning(f"Cosine similarity calculation failed: {e}")
        return 0.0


def jaccard_similarity(text1: str, text2: str) -> float:
    """Calculate Jaccard similarity between two texts."""
    try:
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    except Exception as e:
        logger.warning(f"Jaccard similarity calculation failed: {e}")
        return 0.0


def length_similarity_penalty(len1: int, len2: int) -> float:
    """Apply penalty for documents with very different lengths."""
    try:
        if len1 == 0 or len2 == 0:
            return 0.0
        
        ratio = min(len1, len2) / max(len1, len2)
        return ratio
    except Exception as e:
        logger.warning(f"Length similarity penalty calculation failed: {e}")
        return 0.0


def calculate_similarity_matrix_vectorized(embeddings: List[np.ndarray]) -> np.ndarray:
    """
    Ultra-fast vectorized cosine similarity calculation using sklearn.
    """
    # Convert to numpy matrix for vectorized operations
    embedding_matrix = np.array(embeddings)
    
    # Single vectorized operation instead of nested loops
    similarity_matrix = sklearn_cosine_similarity(embedding_matrix)
    
    return similarity_matrix


def calculate_jaccard_similarity_batch(processed_texts: List[str]) -> np.ndarray:
    """
    Batch Jaccard similarity calculation with early termination.
    """
    n = len(processed_texts)
    jaccard_matrix = np.zeros((n, n))
    
    # Pre-tokenize all texts
    word_sets = [set(text.split()) for text in processed_texts]
    
    # Vectorized similarity calculation
    for i in range(n):
        for j in range(i + 1, n):
            intersection = len(word_sets[i] & word_sets[j])
            union = len(word_sets[i] | word_sets[j])
            
            if union > 0:
                jaccard_sim = intersection / union
                jaccard_matrix[i][j] = jaccard_sim
                jaccard_matrix[j][i] = jaccard_sim
    
    np.fill_diagonal(jaccard_matrix, 1.0)
    return jaccard_matrix


def calculate_document_similarity_optimized(doc_texts: List[str], doc_names: List[str]) -> np.ndarray:
    """
    Optimized similarity calculation with vectorized operations and early termination.
    """
    start_time = time.time()
    
    if len(doc_texts) < 2:
        return np.array([[1.0]])
    
    try:
        # Fast preprocessing with caching
        processed_texts = [preprocess_text_fast(text) for text in doc_texts]
        preprocess_time = time.time() - start_time
        
        # Batch embedding generation with progress tracking
        embedding_start = time.time()
        embeddings = embedding_service.embed_documents(doc_texts)
        embedding_time = time.time() - embedding_start
        
        # Vectorized cosine similarity (much faster than loops)
        cos_sim_start = time.time()
        cosine_matrix = calculate_similarity_matrix_vectorized(embeddings)
        cos_sim_time = time.time() - cos_sim_start
        
        # Batch Jaccard similarity
        jaccard_start = time.time()
        jaccard_matrix = calculate_jaccard_similarity_batch(processed_texts)
        jaccard_time = time.time() - jaccard_start
        
        # Vectorized length penalty calculation
        lengths = np.array([len(text) for text in processed_texts])
        length_matrix = np.minimum(lengths[:, None], lengths) / np.maximum(lengths[:, None], lengths)
        np.fill_diagonal(length_matrix, 1.0)
        
        # Vectorized combined similarity
        similarity_matrix = (
            cosine_matrix * 0.6 + 
            jaccard_matrix * 0.3 + 
            length_matrix * 0.1
        )
        
        total_time = time.time() - start_time

        
        return similarity_matrix
        
    except Exception as e:
        logger.error(f"Error in optimized similarity calculation: {e}")
        # Fallback to original method
        return calculate_document_similarity_improved_fallback(doc_texts, doc_names)


def calculate_document_similarity_improved(doc_texts: List[str], doc_names: List[str]) -> np.ndarray:
    """
    Main improved similarity function - uses optimized version.
    """
    return calculate_document_similarity_optimized(doc_texts, doc_names)


def calculate_document_similarity_improved_fallback(doc_texts: List[str], doc_names: List[str]) -> np.ndarray:
    """
    Fallback version of the original improved method.
    """
    if len(doc_texts) < 2:
        return np.array([[1.0]])
    
    try:
        # Preprocess and clean document texts
        processed_texts = []
        for text in doc_texts:
            # Remove common stop words and normalize
            cleaned = preprocess_text(text)
            processed_texts.append(cleaned)
        
        # Get embeddings for processed documents
        embeddings = embedding_service.embed_documents(doc_texts)
        
        # Calculate similarity matrix with multiple metrics
        similarity_matrix = np.zeros((len(embeddings), len(embeddings)))
        
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                # Cosine similarity
                cos_sim = cosine_similarity(embeddings[i], embeddings[j])
                
                # Jaccard similarity on key terms
                jaccard_sim = jaccard_similarity(processed_texts[i], processed_texts[j])
                
                # Length similarity penalty
                length_penalty = length_similarity_penalty(len(processed_texts[i]), len(processed_texts[j]))
                
                # Combined similarity score
                combined_sim = (cos_sim * 0.6 + jaccard_sim * 0.3 + length_penalty * 0.1)
                
                similarity_matrix[i][j] = combined_sim
                similarity_matrix[j][i] = combined_sim
        
        # Set diagonal to 1.0
        np.fill_diagonal(similarity_matrix, 1.0)
        

        return similarity_matrix
        
    except Exception as e:
        logger.error(f"Error calculating improved document similarity: {e}")
        # Final fallback to basic method
        return calculate_document_similarity(doc_texts)


def calculate_adaptive_thresholds(similarity_matrix: np.ndarray, doc_names: List[str]) -> Dict[str, float]:
    """
    Calculate adaptive similarity thresholds based on document characteristics.
    Updated thresholds: 0.8+ = strong, 0.6+ = moderate, 0.4+ = weak but related
    """
    try:
        # Calculate statistics of similarity scores
        similarities = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
        
        if len(similarities) == 0:
            return {
                "highly_similar": 0.8,
                "similar_to": 0.6,
                "related_to": 0.4
            }
        
        # Use fixed relaxed thresholds as specified by user
        # User specified: "0.8 is strong similarity, 0.4 is minimum"
        highly_similar_threshold = 0.8
        similar_threshold = 0.6  
        related_threshold = 0.4   # This is the minimum as requested
        
        # Log similarity statistics for debugging
        sim_mean = np.mean(similarities)
        sim_max = np.max(similarities)
        sim_min = np.min(similarities)
        

        
        return {
            "highly_similar": float(highly_similar_threshold),
            "similar_to": float(similar_threshold),
            "related_to": float(related_threshold)
        }
    except Exception as e:
        logger.error(f"Error calculating adaptive thresholds: {e}")
        return {
            "highly_similar": 0.8,
            "similar_to": 0.6,
            "related_to": 0.4
        }


def passes_quality_filters(doc1_name: str, doc2_name: str, similarity: float) -> bool:
    """
    Apply quality filters to prevent spurious relationships.
    Relaxed filters for better edge creation.
    """
    try:
        # Filter out very short documents (relaxed)
        if len(doc1_name) < 5 or len(doc2_name) < 5:
            return False
        
        # Filter out documents with very similar names (likely duplicates)
        name_similarity = jaccard_similarity(doc1_name.lower(), doc2_name.lower())
        if name_similarity > 0.9 and similarity < 0.7:  # Relaxed duplicate detection
            return False
        
        # Relaxed minimum threshold - now allows 0.4+ relationships
        if similarity < 0.35:  # Only filter out very weak relationships
            return False
        
        return True
    except Exception as e:
        logger.warning(f"Quality filter check failed: {e}")
        return True  # Default to allowing the relationship


def create_quality_edges(doc_ids: List[str], doc_names: List[str], 
                        similarity_matrix: np.ndarray, thresholds: Dict[str, float],
                        doc_contents: Dict[str, str]) -> List[GraphEdge]:
    """
    Create edges with quality filtering and better relationship categorization.
    """
    edges = []
    
    for i in range(len(doc_ids)):
        for j in range(i + 1, len(doc_ids)):
            similarity = similarity_matrix[i][j]
            
            # Add small tolerance buffer to prevent precision issues
            tolerance = 0.001
            effective_related_threshold = thresholds["related_to"] - tolerance
            
            # Determine relationship type based on adaptive thresholds
            # Use frontend-compatible relationship names
            if similarity >= (thresholds["highly_similar"] - tolerance):
                relationship = "highly_similar"
                edge_weight = similarity
            elif similarity >= (thresholds["similar_to"] - tolerance):
                relationship = "similar_to"  # Frontend expects this exact name
                edge_weight = similarity
            elif similarity >= effective_related_threshold:
                relationship = "related_to"  # Frontend expects this exact name
                edge_weight = similarity
            else:
                continue  # Skip weak relationships
            
            # Additional quality checks
            if not passes_quality_filters(doc_names[i], doc_names[j], similarity):
                continue
            
            # Generate mathematical relationship summary (no API calls)
            summary = generate_mathematical_summary(
                doc_names[i], 
                doc_names[j], 
                similarity,
                relationship
            )
            
            new_edge = GraphEdge(
                source=doc_ids[i],
                target=doc_ids[j],
                relationship=relationship,
                weight=float(edge_weight),
                properties={
                    "summary": summary, 
                    "raw_similarity": float(similarity),
                    "similarity_score": float(similarity),  # Alternative property name
                    "edge_type": relationship,
                    "strength": float(edge_weight)
                }
            )
            edges.append(new_edge)
    
    return edges


def hierarchical_clustering_fallback(G: nx.Graph) -> Dict[str, int]:
    """
    Fallback community detection using hierarchical clustering.
    """
    try:
        # Create similarity matrix for clustering
        nodes_list = list(G.nodes())
        similarity_matrix = np.zeros((len(nodes_list), len(nodes_list)))
        
        for i, node1 in enumerate(nodes_list):
            for j, node2 in enumerate(nodes_list):
                if G.has_edge(node1, node2):
                    similarity_matrix[i][j] = G[node1][node2]['weight']
                else:
                    similarity_matrix[i][j] = 0.0
        
        # Determine optimal number of clusters
        n_clusters = min(len(nodes_list), max(2, len(nodes_list) // 3))
        
        # Perform hierarchical clustering
        try:
            from sklearn.cluster import AgglomerativeClustering
            
            # Handle sklearn version compatibility
            try:
                clustering = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    metric='precomputed',  # Updated parameter name
                    linkage='average'
                )
            except TypeError:
                # Fallback for older sklearn versions
                clustering = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    affinity='precomputed',
                    linkage='average'
                )
            
            cluster_labels = clustering.fit_predict(similarity_matrix)
            
            # Map back to node IDs
            communities = {}
            for i, node_id in enumerate(nodes_list):
                communities[node_id] = int(cluster_labels[i])
            
            logger.info(f"Hierarchical clustering created {n_clusters} communities")
            return communities
            
        except ImportError:
            logger.warning("scikit-learn not available, using simple clustering")
            raise Exception("scikit-learn not available")
        
    except Exception as e:
        logger.error(f"Hierarchical clustering fallback failed: {e}")
        # Return simple connected components as last resort
        communities = {}
        community_id = 0
        visited = set()
        
        for node in G.nodes:
            if node not in visited:
                component = nx.node_connected_component(G, node)
                for n in component:
                    communities[n] = community_id
                community_id += 1
                visited.update(component)
        
        logger.info(f"Simple clustering created {community_id} communities")
        return communities


async def generate_relationship_summary_optimized(doc1_content: str, doc2_content: str, doc1_name: str, doc2_name: str, similarity_score: float) -> str:
    """
    Generate a concise summary of how two documents are related using optimized Gemini service.
    
    Args:
        doc1_content: Content of first document
        doc2_content: Content of second document
        doc1_name: Name of first document
        doc2_name: Name of second document
        similarity_score: Similarity score between documents
        
    Returns:
        Concise relationship summary
    """
    try:
        # Import optimized services
        from ..services.gemini_optimizer import gemini_service
        from ..utils.prompt_optimizer import PromptOptimizer
        
        # Create content hashes for caching
        doc1_hash = gemini_service._hash_content(doc1_content)
        doc2_hash = gemini_service._hash_content(doc2_content)
        
        # Check cache first
        cached_summary = gemini_service.get_cached_relationship_summary(doc1_hash, doc2_hash)
        if cached_summary:
            logger.info(f"📋 Using cached relationship summary for {doc1_name} ↔ {doc2_name}")
            return cached_summary
        
        # Create optimized prompt
        prompt = PromptOptimizer.create_relationship_prompt(
            doc1_content, doc2_content, similarity_score
        )
        
        # Generate new summary using optimized service
        summary = await gemini_service._async_generate_content(prompt)
        
        if summary and not summary.startswith("Error:"):
            # Cache the successful response
            gemini_service.cache_relationship_summary(doc1_hash, doc2_hash, summary)
            logger.info(f"✨ Generated new relationship summary for {doc1_name} ↔ {doc2_name}")
            return summary
        else:
            raise Exception(f"Gemini generation failed: {summary}")
        
    except Exception as e:
        logger.warning(f"Optimized relationship summary failed: {e}")
        # Enhanced fallback summary
        relationship_type = "highly related" if similarity_score > 0.9 else "similar" if similarity_score > 0.8 else "related"
        
        # Try to infer topic from document names
        topic_hint = ""
        common_keywords = ["neural", "network", "synthesis", "design", "analysis", "system", "approach", "method"]
        for keyword in common_keywords:
            if keyword.lower() in doc1_name.lower() and keyword.lower() in doc2_name.lower():
                topic_hint = f" both focusing on {keyword}-related topics"
                break
        
        return f"These documents are {relationship_type} with {similarity_score:.0%} similarity{topic_hint}. They likely share common methodologies, theoretical frameworks, or application domains based on semantic analysis."

# Keep the old function for backward compatibility
def generate_relationship_summary(doc1_content: str, doc2_content: str, doc1_name: str, doc2_name: str, similarity_score: float) -> str:
    """Legacy function - use generate_relationship_summary_optimized instead."""
    import asyncio
    try:
        # Try to run the async function in the current event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're in an async context, we can't use run_until_complete
            # Fall back to the old implementation
            return generate_relationship_summary_fallback(doc1_content, doc2_content, doc1_name, doc2_name, similarity_score)
        else:
            return loop.run_until_complete(
                generate_relationship_summary_optimized(doc1_content, doc2_content, doc1_name, doc2_name, similarity_score)
            )
    except:
        # Fall back to the old implementation
        return generate_relationship_summary_fallback(doc1_content, doc2_content, doc1_name, doc2_name, similarity_score)

def generate_relationship_summary_fallback(doc1_content: str, doc2_content: str, doc1_name: str, doc2_name: str, similarity_score: float) -> str:
    """Fallback implementation when async is not available."""
    try:
        # Check if Gemini API key is available
        from ..settings import settings
        if not settings.GEMINI_API_KEY:
            logger.warning("GEMINI_API_KEY not found, using fallback summary")
            raise Exception("No API key")
        
        # Truncate content to avoid token limits
        max_chars = 1000  # Reduced to avoid token limit issues
        doc1_sample = doc1_content[:max_chars] + "..." if len(doc1_content) > max_chars else doc1_content
        doc2_sample = doc2_content[:max_chars] + "..." if len(doc2_content) > max_chars else doc2_content
        
        prompt = f"""Analyze these two documents and explain how they are related in 2-3 sentences:

Document 1: {doc1_name}
Content: {doc1_sample}

Document 2: {doc2_name}  
Content: {doc2_sample}

Similarity: {similarity_score:.0%}

Provide a clear, concise summary focusing on shared topics, themes, or methodologies."""
        
        # Use direct Gemini API call
        import google.generativeai as genai
        
        # Configure Gemini with timeout and safety settings
        genai.configure(api_key=settings.GEMINI_API_KEY)
        model = genai.GenerativeModel(
            'gemini-1.5-flash',
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=200,
                temperature=0.7,
            )
        )
        
        response = model.generate_content(prompt)
        
        logger.debug(f"Gemini response type: {type(response)}")
        logger.debug(f"Gemini response text: {response.text if response else 'No response'}")
        
        if response and response.text:
            summary_text = response.text.strip()
            logger.debug(f"Generated summary: {summary_text}")
            return summary_text
        else:
            raise Exception("Empty response from Gemini")
        
    except Exception as e:
        logger.warning(f"Gemini relationship summary failed: {e}")
        # Enhanced fallback summary
        relationship_type = "highly related" if similarity_score > 0.8 else "similar" if similarity_score > 0.7 else "related"
        
        # Try to infer topic from document names
        topic_hint = ""
        common_keywords = ["neural", "network", "synthesis", "design", "analysis", "system", "approach", "method"]
        for keyword in common_keywords:
            if keyword.lower() in doc1_name.lower() and keyword.lower() in doc2_name.lower():
                topic_hint = f" both focusing on {keyword}-related topics"
                break
        
        return f"These documents are {relationship_type} with {similarity_score:.0%} similarity{topic_hint}. They likely share common methodologies, theoretical frameworks, or application domains based on semantic analysis."


def calculate_document_similarity(doc_texts: List[str]) -> np.ndarray:
    """
    Fast vectorized semantic similarity calculation.
    """
    start_time = time.time()
    
    if len(doc_texts) < 2:
        return np.array([[1.0]])
    
    try:
        # Get embeddings for all documents
        embeddings = embedding_service.embed_documents(doc_texts)
        
        # Use vectorized cosine similarity from sklearn (much faster)
        similarity_matrix = calculate_similarity_matrix_vectorized(embeddings)
        
        calc_time = time.time() - start_time

        
        return similarity_matrix
        
    except Exception as e:
        logger.error(f"Error calculating document similarity: {e}")
        # Return identity matrix as fallback
        return np.eye(len(doc_texts))


def calculate_document_similarity_with_threshold(doc_texts: List[str], min_threshold: float = 0.1) -> np.ndarray:
    """
    Calculate similarity with early termination for very dissimilar documents.
    """
    start_time = time.time()
    
    if len(doc_texts) < 2:
        return np.array([[1.0]])
    
    try:
        # Get embeddings
        embeddings = embedding_service.embed_documents(doc_texts)
        
        # Fast vectorized calculation
        similarity_matrix = calculate_similarity_matrix_vectorized(embeddings)
        
        # Apply threshold - set very low similarities to 0 for performance
        similarity_matrix[similarity_matrix < min_threshold] = 0.0
        
        calc_time = time.time() - start_time

        
        return similarity_matrix
        
    except Exception as e:
        logger.error(f"Error in threshold similarity calculation: {e}")
        return np.eye(len(doc_texts))


def calculate_centrality_metrics(nodes: List[GraphNode], edges: List[GraphEdge]) -> Dict[str, Dict[str, float]]:
    """
    Calculate centrality metrics for all nodes in the graph.
    """
    try:
        # Create NetworkX graph
        G = nx.Graph()
        
        # Add nodes
        for node in nodes:
            if node.type == "document":  # Only analyze document nodes
                G.add_node(node.id, label=node.label)
        
        # Add edges
        for edge in edges:
            if edge.source in G.nodes and edge.target in G.nodes:
                G.add_edge(edge.source, edge.target, weight=edge.weight)
        
        if len(G.nodes) == 0:
            return {}
        
        centrality_metrics = {}
        
        # Calculate different centrality measures
        degree_centrality = nx.degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G, weight='weight')
        closeness_centrality = nx.closeness_centrality(G, distance='weight')
        eigenvector_centrality = nx.eigenvector_centrality(G, weight='weight', max_iter=1000) if len(G.nodes) > 1 else {node: 0 for node in G.nodes}
        
        # Combine metrics
        for node_id in G.nodes:
            centrality_metrics[node_id] = {
                'degree': degree_centrality.get(node_id, 0),
                'betweenness': betweenness_centrality.get(node_id, 0),
                'closeness': closeness_centrality.get(node_id, 0),
                'eigenvector': eigenvector_centrality.get(node_id, 0),
                'composite': (
                    degree_centrality.get(node_id, 0) * 0.3 +
                    betweenness_centrality.get(node_id, 0) * 0.3 +
                    closeness_centrality.get(node_id, 0) * 0.2 +
                    eigenvector_centrality.get(node_id, 0) * 0.2
                )
            }
        
        return centrality_metrics
        
    except Exception as e:
        logger.error(f"Error calculating centrality metrics: {e}")
        return {}


def detect_communities(nodes: List[GraphNode], edges: List[GraphEdge]) -> Dict[str, int]:
    """
    Improved community detection using hierarchical clustering and quality metrics.
    """
    try:
        # Create NetworkX graph
        G = nx.Graph()
        
        # Add document nodes only
        doc_nodes = [node for node in nodes if node.type == "document"]
        for node in doc_nodes:
            G.add_node(node.id)
        
        # Add edges with weights
        for edge in edges:
            if edge.source in G.nodes and edge.target in G.nodes:
                G.add_edge(edge.source, edge.target, weight=edge.weight)
        
        if len(G.nodes) < 2:
            return {node.id: 0 for node in doc_nodes}
        
        # Use Louvain community detection with resolution parameter
        if community_louvain is not None:
            try:
                # Try different resolution parameters for optimal community detection
                best_partition = None
                best_modularity = -1
                
                for resolution in [0.5, 1.0, 1.5, 2.0]:
                    partition = community_louvain.best_partition(G, weight='weight', resolution=resolution)
                    modularity = community_louvain.modularity(partition, G, weight='weight')
                    
                    if modularity > best_modularity:
                        best_modularity = modularity
                        best_partition = partition
                
                if best_partition:
                    logger.info(f"Best community detection: {len(set(best_partition.values()))} communities, "
                              f"modularity: {best_modularity:.3f}")
                    return best_partition
                    
            except Exception as e:
                logger.warning(f"Louvain community detection failed: {e}")
        
        # Fallback to hierarchical clustering
        return hierarchical_clustering_fallback(G)
        
    except Exception as e:
        logger.error(f"Error in improved community detection: {e}")
        return {node.id: 0 for node in nodes if node.type == "document"}


def generate_mathematical_summary(doc1_name: str, doc2_name: str, similarity: float, relationship: str) -> str:
    """
    Generate a mathematical relationship summary without API calls.
    
    Args:
        doc1_name: Name of first document
        doc2_name: Name of second document
        similarity: Similarity score between documents
        relationship: Relationship type (highly_similar, similar_to, related_to)
        
    Returns:
        Mathematical relationship summary
    """
    # Extract key terms from document names
    doc1_terms = set(doc1_name.lower().replace('_', ' ').replace('-', ' ').replace('.pdf', '').split())
    doc2_terms = set(doc2_name.lower().replace('_', ' ').replace('-', ' ').replace('.pdf', '').split())
    
    # Find common terms
    common_terms = doc1_terms.intersection(doc2_terms)
    
    # Filter out common words
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'for', 'design', 'approach', 'method', 'analysis', 'system', 'neural', 'network', 'synthesis', 'high', 'level', 'roadmap', 'ckks', 'fhew', 'tfhe', 'fpga', 'ai', 'prist'}
    meaningful_terms = [term for term in common_terms if term not in stop_words and len(term) > 2]
    
    # Generate relationship description
    if relationship == "highly_similar":
        strength_desc = "highly similar"
        confidence = "strong"
    elif relationship == "similar_to":
        strength_desc = "moderately similar"
        confidence = "moderate"
    else:  # related_to
        strength_desc = "somewhat related"
        confidence = "weak"
    
    # Build summary based on common terms
    if meaningful_terms:
        common_topics = ", ".join(meaningful_terms[:3])  # Limit to 3 terms
        summary = f"Documents are {strength_desc} ({similarity:.1%} similarity) with {confidence} confidence. They share common topics: {common_topics}."
    else:
        # Fallback when no meaningful common terms found
        summary = f"Documents are {strength_desc} ({similarity:.1%} similarity) with {confidence} confidence. They likely share underlying themes or methodologies."
    
    return summary


def assign_node_properties(text: str, node_type: str) -> Dict:
    """
    Assign properties to a graph node based on its text content.
    
    Args:
        text: Node text content
        node_type: Type of node
        
    Returns:
        Dictionary of node properties
    """
    properties = {
        "length": len(text),
        "word_count": len(text.split()) if text else 0
    }
    
    # Basic document properties only
    properties["document_type"] = "research_paper"
    
    return properties


@router.get("/generate/{session_id}", response_model=KnowledgeGraphResponse)
def generate_knowledge_graph(session_id: str, similarity_threshold: float = None):
    """
    Generate a knowledge graph for all documents in a session.
    
    Args:
        session_id: Session to generate graph for
        similarity_threshold: Minimum similarity for document links (default 0.4)
                             0.8+ = strong similarity, 0.6+ = moderate, 0.4+ = weak but related
        
    Returns:
        Knowledge graph with nodes and edges
        
    Similarity Scale:
        - 0.8+: Strong similarity (highly related documents)
        - 0.6-0.8: Moderate similarity (related topics/methods)  
        - 0.4-0.6: Weak similarity (some common themes)
        - <0.4: Not related (filtered out)
    """
    # Import services locally to avoid any import conflicts
    from ..services.storage import storage_service as storage_svc
    from ..services.index import vector_index_service
    
    # Use default threshold if not provided
    if similarity_threshold is None:
        similarity_threshold = settings.DEFAULT_SIMILARITY_THRESHOLD
    
    # Validate session
    session = storage_svc.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=404,
            detail=f"Session not found: {session_id}"
        )
    
    try:
        # Get all documents in session
        documents = storage_svc.get_session_documents(session_id)
        
        if not documents:
            return KnowledgeGraphResponse(
                nodes=[],
                edges=[],
                session_id=session_id,
                graph_stats={"total_nodes": 0, "total_edges": 0},
                message="No documents found in session"
            )
        
        nodes = []
        edges = []
        
        # Collect document texts for similarity calculation
        doc_texts = []
        doc_ids = []
        doc_names = []
        doc_contents = {}
        
        # Process each document
        for doc in documents:
            doc_id = doc['document_id']
            doc_name = doc['filename']
            
            # Get document content using strategic selection
            doc_content = ""
            content_metadata = {}
            
            try:
                # Try strategic content selection first
                from ..services.content_selector import get_strategic_document_content
                
                strategic_content, content_metadata = get_strategic_document_content(
                    session_id, doc_id, doc_name, 
                    strategy=settings.CONTENT_SELECTION_STRATEGY
                )
                
                if strategic_content:
                    doc_content = strategic_content
                    strategy_used = content_metadata.get('strategy_used', 'unknown')
                    quality_score = content_metadata.get('selection_quality', 0.0)
                    logger.info(f"✅ Strategic content for {doc_name}: {len(doc_content)} chars "
                               f"(strategy: {strategy_used}, quality: {quality_score:.2f})")
                else:
                    # Fallback to semantic search
                    sample_chunks = semantic_search(session_id, doc_name, 3)
                    if sample_chunks:
                        doc_content = " ".join([chunk.get('content', '') for chunk in sample_chunks])
                        logger.info(f"✅ Fallback semantic content for {doc_name}: {len(doc_content)} chars")
                        content_metadata = {"strategy_used": "semantic_fallback", "chunks_used": len(sample_chunks)}
                    else:
                        # Enhanced fallback - use document properties for similarity
                        doc_content = f"{doc_name} {doc.get('filename', '')} pages:{doc.get('page_count', 0)}"
                        logger.warning(f"❌ No content found for {doc_name} in session {session_id}, using basic fallback")
                        content_metadata = {"strategy_used": "basic_fallback", "reason": "no_semantic_results"}
                        
                        # Debug: Check if there are chunks in storage for this document
                        try:
                            chunks = storage_svc.get_document_chunks(doc_id)
                            if chunks:
                                logger.warning(f"🔍 Found {len(chunks)} chunks in storage for {doc_name}, but no index access")
                                # Use chunks directly as fallback
                                doc_content = " ".join([chunk.get('content', '')[:500] for chunk in chunks[:3]])
                                logger.info(f"🔧 Using storage chunks directly: {len(doc_content)} chars")
                                content_metadata.update({"strategy_used": "storage_fallback", "chunks_used": len(chunks)})
                            else:
                                logger.warning(f"🔍 No chunks found in storage for document {doc_id}")
                        except Exception as storage_e:
                            logger.error(f"🔍 Storage check failed for {doc_id}: {storage_e}")
                        
            except Exception as e:
                logger.error(f"Strategic content selection failed for {doc_id}: {e}")
                doc_content = doc_name
                content_metadata = {"strategy_used": "error_fallback", "error": str(e)}
            
            doc_texts.append(doc_content)
            doc_ids.append(doc_id)
            doc_names.append(doc_name)
            doc_contents[doc_id] = doc_content
            
            # Create document node
            doc_properties = assign_node_properties(doc_content, "document")
            doc_properties.update({
                "filename": doc_name,
                "pages": doc.get('page_count', 0),
                "uploaded_at": doc.get('upload_time'),
                "content_strategy": content_metadata.get('strategy_used', 'unknown'),
                "content_quality": content_metadata.get('selection_quality', 0.0),
                "content_length": len(doc_content)
            })
            
            doc_node = GraphNode(
                id=doc_id,
                label=doc_name,
                type="document",
                properties=doc_properties
            )
            nodes.append(doc_node)
            
            # Only process document nodes - no concept/entity extraction
        
        # Calculate document similarities using optimized methods
        if len(doc_texts) > 1:
            try:
                similarity_start_time = time.time()
                
                # Choose similarity calculation method based on document count and settings
                if settings.SIMILARITY_CALCULATION_MODE == "optimized":
                    similarity_matrix = calculate_document_similarity_optimized(doc_texts, doc_names)
                elif settings.SIMILARITY_CALCULATION_MODE == "threshold" or len(doc_texts) > settings.MAX_DOCUMENTS_FOR_FULL_GRAPH:
                    similarity_matrix = calculate_document_similarity_with_threshold(doc_texts, settings.SIMILARITY_MIN_THRESHOLD)
                else:
                    similarity_matrix = calculate_document_similarity_improved_fallback(doc_texts, doc_names)
                
                similarity_total_time = time.time() - similarity_start_time
                
                # Calculate adaptive thresholds
                adaptive_thresholds = calculate_adaptive_thresholds(similarity_matrix, doc_names)
                
                # Override with user threshold if provided and higher
                if similarity_threshold > adaptive_thresholds["related_to"]:
                    adaptive_thresholds["related_to"] = similarity_threshold
                
                # Create quality edges with filtering
                edges = create_quality_edges(doc_ids, doc_names, similarity_matrix, adaptive_thresholds, doc_contents)
                
            except Exception as e:
                logger.error(f"Error calculating improved document similarities: {e}")
                # Fallback to original method
                try:
                    similarity_matrix = calculate_document_similarity(doc_texts)
                    
                    for i in range(len(doc_ids)):
                        for j in range(i + 1, len(doc_ids)):
                            similarity = similarity_matrix[i][j]
                            if similarity > similarity_threshold:
                                # Categorize similarity level for better edge semantics
                                if similarity > 0.9:
                                    relationship = "highly_similar"
                                elif similarity > 0.8:
                                    relationship = "similar_to"
                                else:
                                    relationship = "related_to"
                                
                                # Generate mathematical relationship summary (no API calls)
                                summary = generate_mathematical_summary(
                                    doc_names[i], 
                                    doc_names[j], 
                                    similarity,
                                    relationship
                                )
                                
                                edges.append(GraphEdge(
                                    source=doc_ids[i],
                                    target=doc_ids[j],
                                    relationship=relationship,
                                    weight=float(similarity),
                                    properties={"summary": summary}
                                ))
                except Exception as fallback_error:
                    logger.error(f"Fallback similarity calculation also failed: {fallback_error}")
                    edges = []
        
        # Only document-to-document relationships based on similarity
        
        # Calculate centrality metrics
        centrality_metrics = calculate_centrality_metrics(nodes, edges)
        
        # Detect communities
        communities = detect_communities(nodes, edges)
        
        # Update node properties with centrality and community info
        for node in nodes:
            if node.id in centrality_metrics:
                node.properties.update({
                    'centrality_degree': centrality_metrics[node.id]['degree'],
                    'centrality_betweenness': centrality_metrics[node.id]['betweenness'],
                    'centrality_closeness': centrality_metrics[node.id]['closeness'],
                    'centrality_composite': centrality_metrics[node.id]['composite']
                })
            
            if node.id in communities:
                node.properties['community'] = communities[node.id]
        
        # Calculate graph statistics
        graph_stats = {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "document_nodes": len([n for n in nodes if n.type == "document"]),
            "communities_count": len(set(communities.values())) if communities else 0,
            "centrality_metrics": centrality_metrics,
            "communities": communities
        }
        
        response = KnowledgeGraphResponse(
            nodes=nodes,
            edges=edges,
            session_id=session_id,
            graph_stats=graph_stats,
            message=f"Generated knowledge graph with {len(nodes)} nodes and {len(edges)} edges"
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Knowledge graph generation error: {e}")
        logger.error(f"Exception type: {type(e).__name__}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Knowledge graph generation failed: {str(e)}"
        )


@router.get("/node/{session_id}/{node_id}")
def get_node_details(session_id: str, node_id: str):
    """
    Get detailed information about a specific node.
    
    Args:
        session_id: Session ID
        node_id: Node ID to get details for
        
    Returns:
        Detailed node information
    """
    # Validate session
    session = storage_service.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=404,
            detail=f"Session not found: {session_id}"
        )
    
    try:
        # Generate graph to find the node
        graph_response = generate_knowledge_graph(session_id)
        
        # Find the requested node
        target_node = None
        for node in graph_response.nodes:
            if node.id == node_id:
                target_node = node
                break
        
        if not target_node:
            raise HTTPException(
                status_code=404,
                detail=f"Node not found: {node_id}"
            )
        
        # Get connected nodes
        connected_nodes = []
        for edge in graph_response.edges:
            if edge.source == node_id:
                connected_nodes.append({
                    "node_id": edge.target,
                    "relationship": edge.relationship,
                    "direction": "outgoing"
                })
            elif edge.target == node_id:
                connected_nodes.append({
                    "node_id": edge.source,
                    "relationship": edge.relationship,
                    "direction": "incoming"
                })
        
        # Get content snippets if it's a document node
        content_snippets = []
        if target_node.type == "document":
            try:
                chunks = semantic_search(session_id, target_node.label, 3)
                content_snippets = [chunk.get('content', '')[:200] + "..." for chunk in chunks]
            except Exception as e:
                logger.warning(f"Could not get content snippets for {node_id}: {e}")
        
        return {
            "node": target_node.dict(),
            "connected_nodes": connected_nodes,
            "connection_count": len(connected_nodes),
            "content_snippets": content_snippets
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Node details error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get node details: {str(e)}"
        )


@router.get("/search/{session_id}")
def search_graph_nodes(session_id: str, query: str, node_types: str = "document,concept,entity"):
    """
    Search for nodes in the knowledge graph.
    
    Args:
        session_id: Session ID
        query: Search query
        node_types: Comma-separated list of node types to search
        
    Returns:
        Matching nodes
    """
    # Validate session
    session = storage_service.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=404,
            detail=f"Session not found: {session_id}"
        )
    
    try:
        # Generate graph
        graph_response = generate_knowledge_graph(session_id)
        
        # Filter by node types
        allowed_types = set(node_types.split(","))
        
        # Search nodes
        query_lower = query.lower()
        matching_nodes = []
        
        for node in graph_response.nodes:
            if node.type in allowed_types:
                # Check if query matches label or properties
                if (query_lower in node.label.lower() or 
                    any(query_lower in str(prop_value).lower() for prop_value in node.properties.values())):
                    matching_nodes.append(node)
        
        return {
            "query": query,
            "node_types_searched": list(allowed_types),
            "matching_nodes": [node.dict() for node in matching_nodes],
            "result_count": len(matching_nodes)
        }
        
    except Exception as e:
        logger.error(f"Graph search error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Graph search failed: {str(e)}"
        )


@router.get("/stats/{session_id}")
async def get_graph_statistics(session_id: str):
    """
    Get statistical information about the knowledge graph.
    
    Args:
        session_id: Session ID
        
    Returns:
        Graph statistics and metrics
    """
    # Validate session
    session = storage_service.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=404,
            detail=f"Session not found: {session_id}"
        )
    
    try:
        # Generate graph
        graph_response = generate_knowledge_graph(session_id)
        
        # Calculate detailed statistics
        node_type_counts = defaultdict(int)
        relationship_type_counts = defaultdict(int)
        node_degree = defaultdict(int)
        
        for node in graph_response.nodes:
            node_type_counts[node.type] += 1
        
        for edge in graph_response.edges:
            relationship_type_counts[edge.relationship] += 1
            node_degree[edge.source] += 1
            node_degree[edge.target] += 1
        
        # Find most connected nodes
        most_connected = sorted(node_degree.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "session_id": session_id,
            "total_stats": graph_response.graph_stats,
            "node_types": dict(node_type_counts),
            "relationship_types": dict(relationship_type_counts),
            "most_connected_nodes": [
                {"node_id": node_id, "connections": count} 
                for node_id, count in most_connected
            ],
            "graph_density": len(graph_response.edges) / max(len(graph_response.nodes) * (len(graph_response.nodes) - 1) / 2, 1)
        }
        
    except Exception as e:
        logger.error(f"Graph statistics error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get graph statistics: {str(e)}"
        )


@router.get("/relationship/{session_id}")
def get_relationship_details(
    session_id: str, 
    source_id: str = Query(..., description="Source node ID"),
    target_id: str = Query(..., description="Target node ID")
):
    """
    Get detailed information about a relationship between two nodes.
    
    Args:
        session_id: Session ID
        source_id: Source node ID
        target_id: Target node ID
        
    Returns:
        Detailed relationship information
    """
    logger.info(f"Relationship request: session={session_id}, source={source_id}, target={target_id}")
    
    # Validate session
    session = storage_service.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=404,
            detail=f"Session not found: {session_id}"
        )
    
    try:
        # Generate graph to find the relationship
        graph_response = generate_knowledge_graph(session_id)
        
        # Find the relationship edge
        target_edge = None
        for edge in graph_response.edges:
            if (edge.source == source_id and edge.target == target_id) or \
               (edge.source == target_id and edge.target == source_id):
                target_edge = edge
                break
        
        if not target_edge:
            raise HTTPException(
                status_code=404,
                detail=f"Relationship not found between {source_id} and {target_id}"
            )
        
        # Find the nodes
        source_node = None
        target_node = None
        for node in graph_response.nodes:
            if node.id == target_edge.source:
                source_node = node
            elif node.id == target_edge.target:
                target_node = node
        
        return {
            "relationship": target_edge.dict(),
            "source_node": source_node.dict() if source_node else None,
            "target_node": target_node.dict() if target_node else None,
            "summary": target_edge.properties.get("summary", "No summary available"),
            "strength": target_edge.weight,
            "type": target_edge.relationship
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Relationship details error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get relationship details: {str(e)}"
        )


@router.get("/cache/stats/{session_id}")
async def get_cache_statistics(session_id: str):
    """
    Get cache statistics for the knowledge graph system.
    
    Args:
        session_id: Session ID (for consistency with other endpoints)
        
    Returns:
        Cache statistics and performance metrics
    """
    try:
        from ..services.gemini_optimizer import gemini_service
        
        cache_stats = gemini_service.get_cache_stats()
        
        return {
            "session_id": session_id,
            "cache_statistics": cache_stats,
            "message": "Cache statistics retrieved successfully"
        }
        
    except Exception as e:
        logger.error(f"Cache statistics error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get cache statistics: {str(e)}"
        )


@router.post("/cache/clear/{session_id}")
async def clear_cache(session_id: str, clear_type: str = "expired"):
    """
    Clear cache for the knowledge graph system.
    
    Args:
        session_id: Session ID (for consistency with other endpoints)
        clear_type: Type of cache clearing ("expired", "all", "relationships", "pdf_analysis")
        
    Returns:
        Cache clearing results
    """
    try:
        from ..services.gemini_optimizer import gemini_service
        
        if clear_type == "expired":
            cleared_count = gemini_service.clear_expired_cache()
            message = f"Cleared {cleared_count} expired cache files"
        elif clear_type == "all":
            # Clear all cache files
            cache_files = list(gemini_service.cache_dir.glob("*.pkl"))
            for cache_file in cache_files:
                cache_file.unlink()
            cleared_count = len(cache_files)
            message = f"Cleared all {cleared_count} cache files"
        elif clear_type == "relationships":
            # Clear only relationship cache files
            cache_files = list(gemini_service.cache_dir.glob("relationship_*.pkl"))
            for cache_file in cache_files:
                cache_file.unlink()
            cleared_count = len(cache_files)
            message = f"Cleared {cleared_count} relationship cache files"
        elif clear_type == "pdf_analysis":
            # Clear only PDF analysis cache files
            cache_files = list(gemini_service.cache_dir.glob("pdf_analysis_*.pkl"))
            for cache_file in cache_files:
                cache_file.unlink()
            cleared_count = len(cache_files)
            message = f"Cleared {cleared_count} PDF analysis cache files"
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid clear_type: {clear_type}. Use 'expired', 'all', 'relationships', or 'pdf_analysis'"
            )
        
        return {
            "session_id": session_id,
            "clear_type": clear_type,
            "cleared_count": cleared_count,
            "message": message
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Cache clearing error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear cache: {str(e)}"
        )


@router.get("/debug/session-info/{session_id}")
async def debug_session_info(session_id: str):
    """
    Debug endpoint to check session data consistency.
    """
    try:
        from ..services.index import vector_index_service
        
        # Check session
        session = storage_service.get_session(session_id)
        
        # Check documents
        documents = storage_service.get_session_documents(session_id) if session else []
        
        # Check chunks for each document
        document_info = []
        total_chunks = 0
        
        for doc in documents:
            try:
                chunks = storage_service.get_document_chunks(doc['document_id'])
                chunk_count = len(chunks) if chunks else 0
                total_chunks += chunk_count
                
                document_info.append({
                    "document_id": doc['document_id'],
                    "filename": doc['filename'],
                    "status": doc.get('status', 'unknown'),
                    "chunk_count": chunk_count,
                    "has_chunks": chunk_count > 0
                })
            except Exception as e:
                document_info.append({
                    "document_id": doc['document_id'],
                    "filename": doc['filename'],
                    "status": doc.get('status', 'unknown'),
                    "chunk_count": 0,
                    "has_chunks": False,
                    "error": str(e)
                })
        
        # Check if index exists
        index_exists = vector_index_service._load_session_index(session_id)
        
        # Get index stats if exists
        index_info = {"exists": False}
        if index_exists and session_id in vector_index_service._session_indices:
            index = vector_index_service._session_indices[session_id]
            index_info = {
                "exists": True,
                "total_vectors": index.ntotal,
                "dimension": index.d
            }
        
        return {
            "session_id": session_id,
            "session_exists": session is not None,
            "document_count": len(documents),
            "total_chunks": total_chunks,
            "index_info": index_info,
            "documents": document_info,
            "diagnosis": {
                "has_session": session is not None,
                "has_documents": len(documents) > 0,
                "has_chunks": total_chunks > 0,
                "has_index": index_info["exists"],
                "ready_for_graph": all([
                    session is not None,
                    len(documents) > 0,
                    total_chunks > 0,
                    index_info["exists"]
                ])
            }
        }
        
    except Exception as e:
        logger.error(f"Session debug error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get session info: {str(e)}"
        )


@router.post("/debug/rebuild-index/{session_id}")
async def rebuild_session_index(session_id: str):
    """
    Debug endpoint to rebuild the vector index for a session.
    Use this when graphs show 'No index found' errors.
    """
    try:
        from ..services.index import vector_index_service
        
        # Validate session
        session = storage_service.get_session(session_id)
        if not session:
            raise HTTPException(
                status_code=404,
                detail=f"Session not found: {session_id}"
            )
        
        # Get all documents in session
        documents = storage_service.get_session_documents(session_id)
        
        if not documents:
            return {
                "session_id": session_id,
                "status": "no_documents",
                "message": "No documents found in session to index"
            }
        
        # Get chunks for each document
        all_chunks = []
        for doc in documents:
            try:
                chunks = storage_service.get_document_chunks(doc['document_id'])
                all_chunks.extend(chunks)
            except Exception as e:
                logger.warning(f"Could not get chunks for document {doc['document_id']}: {e}")
        
        if not all_chunks:
            return {
                "session_id": session_id,
                "status": "no_chunks",
                "message": "No chunks found to index"
            }
        
        # Convert to Document objects
        from langchain.schema import Document
        documents_to_index = []
        for chunk in all_chunks:
            documents_to_index.append(Document(
                page_content=chunk.get('content', ''),
                metadata=chunk.get('metadata', {})
            ))
        
        # Rebuild index
        success = vector_index_service.add_documents(session_id, documents_to_index)
        
        return {
            "session_id": session_id,
            "status": "success" if success else "failed",
            "documents_processed": len(documents),
            "chunks_indexed": len(all_chunks),
            "message": f"Index {'rebuilt successfully' if success else 'rebuild failed'}"
        }
        
    except Exception as e:
        logger.error(f"Index rebuild error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to rebuild index: {str(e)}"
        )


@router.get("/cache/health/{session_id}")
async def get_cache_health(session_id: str):
    """
    Get cache health and performance metrics.
    
    Args:
        session_id: Session ID (for consistency with other endpoints)
        
    Returns:
        Cache health information
    """
    try:
        from ..services.gemini_optimizer import gemini_service
        
        cache_stats = gemini_service.get_cache_stats()
        
        # Calculate cache health metrics
        total_files = cache_stats.get("total_files", 0)
        total_size_mb = cache_stats.get("total_size_mb", 0)
        
        # Health indicators
        health_status = "healthy"
        if total_size_mb > 100:  # More than 100MB
            health_status = "warning"
        if total_size_mb > 500:  # More than 500MB
            health_status = "critical"
        
        # Performance recommendations
        recommendations = []
        if total_files > 1000:
            recommendations.append("Consider clearing old cache files")
        if total_size_mb > 100:
            recommendations.append("Cache size is large, consider cleanup")
        
        return {
            "session_id": session_id,
            "health_status": health_status,
            "cache_statistics": cache_stats,
            "recommendations": recommendations,
            "message": "Cache health check completed"
        }
        
    except Exception as e:
        logger.error(f"Cache health check error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to check cache health: {str(e)}"
        )
