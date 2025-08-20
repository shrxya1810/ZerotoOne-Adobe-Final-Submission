"""
Optimized Gemini service with caching, batching, and connection pooling.
Specifically designed for PDF analysis and knowledge graph generation.
"""
import asyncio
import hashlib
import json
import time
import pickle
from typing import List, Dict, Any, Optional, Tuple
from functools import lru_cache
import google.generativeai as genai
from ..settings import settings
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class OptimizedGeminiService:
    def __init__(self):
        self.model = None
        self.cache_dir = Path("data/cache/gemini")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.request_queue = []
        self.batch_size = 5  # Process multiple requests together
        self.cache_ttl = 3600 * 24 * 7  # 1 week cache for PDF analysis
        
        if settings.GEMINI_API_KEY:
            genai.configure(api_key=settings.GEMINI_API_KEY)
            self.model = genai.GenerativeModel(settings.GEMINI_MODEL)
            logger.info("✅ Optimized Gemini service initialized")
        else:
            logger.warning("⚠️ Gemini API key not configured")
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get file path for cached content."""
        return self.cache_dir / f"{cache_key}.pkl"
    
    def _hash_content(self, content: str, max_length: int = 1000) -> str:
        """Create hash for content caching with length normalization."""
        # Normalize content length for consistent hashing
        normalized = content[:max_length] + f"__LEN_{len(content)}"
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def get_cached_pdf_analysis(self, pdf_content_hash: str) -> Optional[Dict]:
        """Get cached PDF analysis results."""
        cache_path = self._get_cache_path(f"pdf_analysis_{pdf_content_hash}")
        
        if cache_path.exists():
            try:
                # Check if cache is still valid
                if time.time() - cache_path.stat().st_mtime < self.cache_ttl:
                    with open(cache_path, 'rb') as f:
                        cached_data = pickle.load(f)
                        logger.info(f"📋 Cache hit for PDF analysis: {pdf_content_hash[:8]}...")
                        return cached_data
                else:
                    # Cache expired, remove it
                    cache_path.unlink()
                    logger.debug(f"🗑️ Expired cache removed: {pdf_content_hash[:8]}...")
            except Exception as e:
                logger.warning(f"Cache read failed: {e}")
                cache_path.unlink(missing_ok=True)
        
        return None
    
    def cache_pdf_analysis(self, pdf_content_hash: str, analysis_data: Dict):
        """Cache PDF analysis results."""
        try:
            cache_path = self._get_cache_path(f"pdf_analysis_{pdf_content_hash}")
            
            # Ensure cache directory exists
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(cache_path, 'wb') as f:
                pickle.dump(analysis_data, f)
            
            logger.info(f"💾 Cached PDF analysis: {pdf_content_hash[:8]}...")
            
        except Exception as e:
            logger.error(f"Failed to cache PDF analysis: {e}")
    
    def get_cached_relationship_summary(self, doc1_hash: str, doc2_hash: str) -> Optional[str]:
        """Get cached relationship summary between two documents."""
        # Create combined hash for the pair
        pair_hash = hashlib.md5(f"{doc1_hash}_{doc2_hash}".encode()).hexdigest()
        cache_path = self._get_cache_path(f"relationship_{pair_hash}")
        
        if cache_path.exists():
            try:
                if time.time() - cache_path.stat().st_mtime < self.cache_ttl:
                    with open(cache_path, 'rb') as f:
                        cached_summary = pickle.load(f)
                        logger.info(f"📋 Cache hit for relationship: {doc1_hash[:8]} ↔ {doc2_hash[:8]}")
                        return cached_summary
                else:
                    cache_path.unlink()
            except Exception as e:
                logger.warning(f"Relationship cache read failed: {e}")
                cache_path.unlink(missing_ok=True)
        
        return None
    
    def cache_relationship_summary(self, doc1_hash: str, doc2_hash: str, summary: str):
        """Cache relationship summary between two documents."""
        try:
            pair_hash = hashlib.md5(f"{doc1_hash}_{doc2_hash}".encode()).hexdigest()
            cache_path = self._get_cache_path(f"relationship_{pair_hash}")
            
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(cache_path, 'wb') as f:
                pickle.dump(summary, f)
            
            logger.info(f"💾 Cached relationship summary: {doc1_hash[:8]} ↔ {doc2_hash[:8]}")
            
        except Exception as e:
            logger.error(f"Failed to cache relationship summary: {e}")
    
    async def batch_generate_content(self, requests: List[Dict]) -> List[str]:
        """Batch multiple similar requests for efficiency."""
        if not self.model:
            return ["AI service not available"] * len(requests)
        
        try:
            # Group similar requests
            batched_results = []
            for i in range(0, len(requests), self.batch_size):
                batch = requests[i:i + self.batch_size]
                
                # Create combined prompt for batch
                combined_prompt = self._create_batch_prompt(batch)
                
                # Single API call for batch
                response = await self._async_generate_content(combined_prompt)
                
                # Parse batch response
                batch_results = self._parse_batch_response(response, len(batch))
                batched_results.extend(batch_results)
            
            return batched_results
            
        except Exception as e:
            logger.error(f"Batch generation failed: {e}")
            return ["Generation failed"] * len(requests)
    
    def _create_batch_prompt(self, requests: List[Dict]) -> str:
        """Create a single prompt for multiple requests."""
        prompt = "Process the following requests and provide responses in JSON format:\n\n"
        
        for i, req in enumerate(requests):
            prompt += f"Request {i+1}:\n"
            prompt += f"Type: {req.get('type', 'unknown')}\n"
            prompt += f"Content: {req.get('content', '')[:200]}...\n\n"
        
        prompt += "Respond with JSON array: [response1, response2, ...]"
        return prompt
    
    def _parse_batch_response(self, response: str, expected_count: int) -> List[str]:
        """Parse batch response into individual results."""
        try:
            results = json.loads(response)
            if isinstance(results, list) and len(results) >= expected_count:
                return results[:expected_count]
        except:
            pass
        
        # Fallback: split response by markers
        return [response] * expected_count
    
    async def _async_generate_content(self, prompt: str) -> str:
        """Async wrapper for Gemini API calls."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            self._sync_generate_content, 
            prompt
        )
    
    def _sync_generate_content(self, prompt: str) -> str:
        """Synchronous Gemini API call with timeout and optimization."""
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=min(500, settings.GEMINI_MAX_TOKENS),  # Reduced for efficiency
                    temperature=settings.GEMINI_TEMPERATURE,
                    timeout=30  # 30 second timeout
                )
            )
            return response.text if response else ""
        except Exception as e:
            logger.error(f"Gemini API call failed: {e}")
            return f"Error: {str(e)}"
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for monitoring."""
        try:
            cache_files = list(self.cache_dir.glob("*.pkl"))
            total_size = sum(f.stat().st_size for f in cache_files)
            
            # Count by type
            pdf_cache = len([f for f in cache_files if "pdf_analysis" in f.name])
            relationship_cache = len([f for f in cache_files if "relationship" in f.name])
            
            return {
                "total_files": len(cache_files),
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "pdf_analysis_cache": pdf_cache,
                "relationship_cache": relationship_cache,
                "cache_ttl_hours": self.cache_ttl / 3600
            }
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {"error": str(e)}
    
    def clear_expired_cache(self) -> int:
        """Clear expired cache files and return count of cleared files."""
        try:
            cleared_count = 0
            current_time = time.time()
            
            for cache_file in self.cache_dir.glob("*.pkl"):
                if current_time - cache_file.stat().st_mtime > self.cache_ttl:
                    cache_file.unlink()
                    cleared_count += 1
            
            if cleared_count > 0:
                logger.info(f"🧹 Cleared {cleared_count} expired cache files")
            
            return cleared_count
            
        except Exception as e:
            logger.error(f"Failed to clear expired cache: {e}")
            return 0

# Global instance
gemini_service = OptimizedGeminiService() 