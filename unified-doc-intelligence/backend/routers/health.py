"""
Health check router for monitoring system status.
"""
import time
from fastapi import APIRouter, Response
from fastapi.responses import JSONResponse

from ..settings import settings
from ..models.schemas import HealthResponse

# Try to import psutil for system metrics (optional)
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

router = APIRouter()

# Track startup time for uptime calculation
_startup_time = time.time()


@router.options("/")
async def health_options():
    """Handle OPTIONS preflight request for health endpoint."""
    return Response(
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        }
    )


@router.get("/", response_model=HealthResponse)
async def health_check():
    """
    Get system health status and capabilities.
    """
    uptime = time.time() - _startup_time
    
    # Check service availability
    services = {}
    capabilities = {}
    
    # Embedding service
    try:
        from ..services.embed import embedding_service
        model_info = embedding_service.get_model_info()
        services["embedding"] = "healthy"
        capabilities["semantic_search"] = True
        # Don't add non-boolean values to capabilities - they go in services
        services["embedding_model"] = model_info["model_name"]
        services["embedding_cache_size"] = str(model_info["cache_size"])
    except Exception as e:
        services["embedding"] = f"error: {str(e)}"
        capabilities["semantic_search"] = False
    
    # Database
    try:
        from ..services.storage import storage_service
        # Simple connectivity test - we need to add get_connection method
        services["database"] = "healthy"
        capabilities["storage"] = True
    except Exception as e:
        services["database"] = f"error: {str(e)}"
        capabilities["storage"] = False
    
    # Gemini AI
    if settings.GEMINI_API_KEY:
        try:
            import google.generativeai as genai
            # Simple test - just check if client is configured
            genai.configure(api_key=settings.GEMINI_API_KEY)
            services["gemini"] = "configured"
            capabilities["ai_insights"] = True
            capabilities["podcast_script"] = True
        except Exception as e:
            services["gemini"] = f"error: {str(e)}"
            capabilities["ai_insights"] = False
            capabilities["podcast_script"] = False
    else:
        services["gemini"] = "not_configured"
        capabilities["ai_insights"] = False
        capabilities["podcast_script"] = False
    
    # TTS Service
    tts_available = bool(settings.AZURE_SPEECH_KEY or settings.GOOGLE_TTS_CREDENTIALS)
    if tts_available:
        try:
            from ..services.tts import tts_service
            provider = tts_service.get_provider()
            services["tts"] = f"healthy ({provider})"
            services["tts_provider"] = provider  # Move to services instead of capabilities
            capabilities["podcast_audio"] = True
        except Exception as e:
            services["tts"] = f"error: {str(e)}"
            capabilities["podcast_audio"] = False
    else:
        services["tts"] = "not_configured"
        capabilities["podcast_audio"] = False
    
    # Vector Index
    try:
        from ..services.index import vector_index_service
        services["vector_index"] = "healthy"
        capabilities["vector_search"] = True
        capabilities["faiss_backend"] = True
    except Exception as e:
        services["vector_index"] = f"error: {str(e)}"
        capabilities["vector_search"] = False
        capabilities["faiss_backend"] = False
    
    # System resources
    if PSUTIL_AVAILABLE:
        try:
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage(settings.UPLOAD_DIR)
            capabilities.update({
                "memory_usage_percent": memory.percent,
                "disk_usage_percent": disk.percent,
                "available_memory_gb": round(memory.available / 1024**3, 2),
                "available_disk_gb": round(disk.free / 1024**3, 2)
            })
        except Exception:
            pass  # System metrics are optional
    
    # Overall status
    all_critical_healthy = all(
        "error" not in status for service, status in services.items()
        if service in ["embedding", "database"]  # Critical services
    )
    
    overall_status = "healthy" if all_critical_healthy else "degraded"
    
    return HealthResponse(
        status=overall_status,
        version=settings.API_VERSION,
        uptime=uptime,
        services=services,
        capabilities=capabilities,
        message=f"System is {overall_status}. Uptime: {uptime:.1f}s"
    )


@router.get("/ping")
async def ping():
    """Simple ping endpoint for basic connectivity."""
    return {"status": "pong", "timestamp": time.time()}


@router.get("/ready")
async def readiness_check():
    """Kubernetes-style readiness check."""
    try:
        # Test critical services
        from ..services.embed import embedding_service
        from ..services.storage import storage_service
        
        # Quick embedding test
        embedding_service.embed_query("test")
        
        # Quick database test - just check if storage service is available
        # We'll add a simple test method to storage service
        
        return {"ready": True, "timestamp": time.time()}
        
    except Exception as e:
        return {"ready": False, "error": str(e), "timestamp": time.time()}


@router.get("/live")
async def liveness_check():
    """Kubernetes-style liveness check."""
    return {"alive": True, "timestamp": time.time()}
