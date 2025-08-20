"""
Search router with semantic, keyword, and hybrid search capabilities.
"""
from fastapi import APIRouter, HTTPException
import logging
from typing import List

from ..services.storage import storage_service, keyword_search
from ..services.index import semantic_search
from ..models.schemas import SearchRequest, SearchResponse, SearchResult, SearchType

logger = logging.getLogger(__name__)

router = APIRouter()


def merge_search_results(semantic_results: List[dict], keyword_results: List[dict], 
                        semantic_weight: float = 0.6) -> List[dict]:
    """
    Merge and rank results from semantic and keyword search.
    
    Args:
        semantic_results: Results from vector search
        keyword_results: Results from FTS search
        semantic_weight: Weight for semantic scores (0.0-1.0)
        
    Returns:
        Merged and ranked results
    """
    # Normalize scores to 0-1 range
    if semantic_results:
        max_semantic = max(r['score'] for r in semantic_results)
        min_semantic = min(r['score'] for r in semantic_results)
        semantic_range = max_semantic - min_semantic
        
        if semantic_range > 0:
            for result in semantic_results:
                result['normalized_score'] = (result['score'] - min_semantic) / semantic_range
        else:
            for result in semantic_results:
                result['normalized_score'] = 1.0
    
    if keyword_results:
        max_keyword = max(abs(r['score']) for r in keyword_results)
        min_keyword = min(abs(r['score']) for r in keyword_results)
        keyword_range = max_keyword - min_keyword
        
        if keyword_range > 0:
            for result in keyword_results:
                result['normalized_score'] = (abs(result['score']) - min_keyword) / keyword_range
        else:
            for result in keyword_results:
                result['normalized_score'] = 1.0
    
    # Create combined result set
    combined = {}
    
    # Add semantic results
    for result in semantic_results:
        chunk_id = result.get('chunk_id', result.get('content', '')[:50])
        combined[chunk_id] = {
            **result,
            'semantic_score': result['normalized_score'],
            'keyword_score': 0.0,
            'source_type': 'semantic'
        }
    
    # Add keyword results
    for result in keyword_results:
        chunk_id = result.get('chunk_id', result.get('content', '')[:50])
        if chunk_id in combined:
            # Update existing result
            combined[chunk_id]['keyword_score'] = result['normalized_score']
            combined[chunk_id]['source_type'] = 'both'
        else:
            # Add new result
            combined[chunk_id] = {
                **result,
                'semantic_score': 0.0,
                'keyword_score': result['normalized_score'],
                'source_type': 'keyword'
            }
    
    # Calculate hybrid scores
    for result in combined.values():
        result['hybrid_score'] = (
            semantic_weight * result['semantic_score'] + 
            (1 - semantic_weight) * result['keyword_score']
        )
    
    # Sort by hybrid score
    sorted_results = sorted(combined.values(), key=lambda x: x['hybrid_score'], reverse=True)
    
    return sorted_results


@router.post("/semantic", response_model=SearchResponse)
async def search_semantic(request: SearchRequest):
    """
    Perform semantic search using vector embeddings.
    
    Uses MPNet or MiniLM embeddings to find semantically similar content
    even when exact keywords don't match.
    
    Args:
        request: Search parameters including query, session_id, and top_k
        
    Returns:
        SearchResponse with ranked results
    """
    # Validate session
    session = storage_service.get_session(request.session_id)
    if not session:
        raise HTTPException(
            status_code=404,
            detail=f"Session not found: {request.session_id}"
        )
    
    try:
        # Perform semantic search
        results = semantic_search(request.session_id, request.query, request.top_k)
        
        # Convert to SearchResult objects
        search_results = []
        for result in results:
            # Convert 0-indexed page number to 1-indexed for user-friendly display
            page_number = result.get('page_number', 0)
            display_page_number = page_number + 1 if isinstance(page_number, int) else 0
            
            search_results.append(SearchResult(
                document=result.get('document', 'Unknown'),
                content=result.get('content', ''),
                page_number=display_page_number,
                score=float(result.get('score', 0.0)),
                metadata=result.get('metadata', {})
            ))
        
        return SearchResponse(
            query=request.query,
            results=search_results,
            total_results=len(search_results),
            search_type=SearchType.SEMANTIC,
            message=f"Found {len(search_results)} semantic matches"
        )
        
    except Exception as e:
        logger.error(f"Semantic search error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Semantic search failed: {str(e)}"
        )


@router.post("/keyword", response_model=SearchResponse)
async def search_keyword(request: SearchRequest):
    """
    Perform keyword search using SQLite FTS5.
    
    Uses full-text search with Porter stemming for exact and fuzzy keyword matching.
    Good for finding specific terms and phrases.
    
    Args:
        request: Search parameters including query, session_id, and top_k
        
    Returns:
        SearchResponse with ranked results
    """
    logger.info(f"🔤 Keyword search request: query='{request.query}', session_id='{request.session_id}', top_k={request.top_k}")
    
    # Validate session
    session = storage_service.get_session(request.session_id)
    if not session:
        logger.error(f"❌ Session not found: {request.session_id}")
        raise HTTPException(
            status_code=404,
            detail=f"Session not found: {request.session_id}"
        )
    
    logger.info(f"✅ Session found: {session.name}")
    
    try:
        # Perform keyword search
        logger.info("🔍 Executing keyword search...")
        results = keyword_search(request.session_id, request.query, request.top_k)
        logger.info(f"📊 Keyword search returned {len(results)} results")
        
        # Convert to SearchResult objects
        search_results = []
        for result in results:
            # Convert 0-indexed page number to 1-indexed for user-friendly display
            page_number = result.get('page_number', 0)
            display_page_number = page_number + 1 if isinstance(page_number, int) else 0
            
            search_results.append(SearchResult(
                document=result.get('document', 'Unknown'),
                content=result.get('content', ''),
                page_number=display_page_number,
                score=float(result.get('score', 0.0)),
                metadata={"search_type": "keyword"}
            ))
        
        logger.info(f"✅ Keyword search completed successfully with {len(search_results)} results")
        
        response = SearchResponse(
            query=request.query,
            results=search_results,
            total_results=len(search_results),
            search_type=SearchType.KEYWORD,
            message=f"Found {len(search_results)} keyword matches"
        )
        
        logger.info(f"📤 Returning keyword search response: {len(response.results)} results")
        return response
        
    except Exception as e:
        logger.error(f"❌ Keyword search error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Keyword search failed: {str(e)}"
        )


@router.post("/hybrid", response_model=SearchResponse)
async def search_hybrid(request: SearchRequest):
    """
    Perform hybrid search combining semantic and keyword results.
    
    Combines the precision of keyword search with the comprehensiveness
    of semantic search for best overall results.
    
    Args:
        request: Search parameters including query, session_id, and top_k
        
    Returns:
        SearchResponse with merged and ranked results
    """
    # Validate session
    session = storage_service.get_session(request.session_id)
    if not session:
        raise HTTPException(
            status_code=404,
            detail=f"Session not found: {request.session_id}"
        )
    
    try:
        # Perform both searches
        semantic_results = semantic_search(request.session_id, request.query, request.top_k)
        keyword_results = keyword_search(request.session_id, request.query, request.top_k)
        
        # Merge results
        merged_results = merge_search_results(semantic_results, keyword_results)
        
        # Take top_k results
        top_results = merged_results[:request.top_k]
        
        # Convert to SearchResult objects
        search_results = []
        for result in top_results:
            # Convert 0-indexed page number to 1-indexed for user-friendly display
            page_number = result.get('page_number', 0)
            display_page_number = page_number + 1 if isinstance(page_number, int) else 0
            
            search_results.append(SearchResult(
                document=result.get('document', 'Unknown'),
                content=result.get('content', ''),
                page_number=display_page_number,
                score=float(result.get('hybrid_score', 0.0)),
                metadata={
                    "search_type": "hybrid",
                    "source_type": result.get('source_type', 'unknown'),
                    "semantic_score": result.get('semantic_score', 0.0),
                    "keyword_score": result.get('keyword_score', 0.0)
                }
            ))
        
        return SearchResponse(
            query=request.query,
            results=search_results,
            total_results=len(search_results),
            search_type=SearchType.HYBRID,
            message=f"Found {len(search_results)} hybrid matches"
        )
        
    except Exception as e:
        logger.error(f"Hybrid search error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Hybrid search failed: {str(e)}"
        )


@router.get("/test/{session_id}")
async def test_search(session_id: str, query: str = "test"):
    """
    Test search functionality for a session.
    
    Args:
        session_id: Session to test
        query: Test query (default: "test")
        
    Returns:
        Search test results and statistics
    """
    # Validate session
    session = storage_service.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=404,
            detail=f"Session not found: {session_id}"
        )
    
    try:
        # Get index stats
        from ..services.index import get_index_stats
        index_stats = get_index_stats(session_id)
        
        # Test semantic search
        semantic_results = semantic_search(session_id, query, 3)
        
        # Test keyword search
        keyword_results = keyword_search(session_id, query, 3)
        
        return {
            "session_id": session_id,
            "test_query": query,
            "index_stats": index_stats,
            "semantic_results_count": len(semantic_results),
            "keyword_results_count": len(keyword_results),
            "semantic_sample": semantic_results[:1],
            "keyword_sample": keyword_results[:1],
            "search_capabilities": {
                "semantic_available": index_stats.get('vector_count', 0) > 0,
                "keyword_available": True,  # Always available with SQLite FTS
                "hybrid_available": True
            }
        }
        
    except Exception as e:
        logger.error(f"Search test error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Search test failed: {str(e)}"
        )
