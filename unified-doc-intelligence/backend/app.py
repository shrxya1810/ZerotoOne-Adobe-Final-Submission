"""
Main FastAPI application for the unified document intelligence backend.
"""
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

from .settings import settings
from .routers import (
    extract_1a, sessions, search, insights, 
    persona, podcast, graph, health, pdfs, upload
)

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup/shutdown tasks."""
    # Startup
    logger.info("🚀 Starting Unified Document Intelligence Backend")
    start_time = time.time()
    
    # Initialize services
    try:
        # Warm up embedding service
        from .services.embed import embedding_service
        embedding_info = embedding_service.get_model_info()
        logger.info(f"✅ Embedding service ready: {embedding_info['model_name']}")
        
        # Initialize Gemini client if available
        if settings.GEMINI_API_KEY:
            try:
                import google.generativeai as genai
                genai.configure(api_key=settings.GEMINI_API_KEY)
                logger.info("✅ Gemini AI client initialized")
            except Exception as e:
                logger.warning(f"Gemini initialization failed: {e}")
        
        # Initialize TTS service if configured
        if settings.AZURE_SPEECH_KEY or settings.GOOGLE_TTS_CREDENTIALS:
            try:
                from .services.tts import tts_service
                logger.info(f"✅ TTS service ready: {tts_service.get_provider()}")
            except Exception as e:
                logger.warning(f"TTS initialization failed: {e}")
        
        # Cleanup expired sessions
        from .services.storage import storage_service
        storage_service.cleanup_expired_sessions()
        
        startup_time = time.time() - start_time
        logger.info(f"🎉 Startup completed in {startup_time:.2f}s")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Unified Document Intelligence Backend")
    
    # Cleanup tasks
    try:
        # Clear embedding cache to free memory
        from .services.embed import embedding_service
        embedding_service.clear_cache()
        logger.info("✅ Cleaned up embedding cache")
        
    except Exception as e:
        logger.error(f"Shutdown cleanup error: {e}")
    
    logger.info("👋 Shutdown completed")


# Create FastAPI app
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description=settings.API_DESCRIPTION,
    lifespan=lifespan,
    debug=settings.DEBUG
)

# Configure CORS
cors_origins = ["http://localhost:3000", "http://localhost:5173"]
if settings.DEBUG:
    cors_origins.append("*")

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["Content-Range", "Accept-Ranges", "Content-Length", "Content-Type"]
)

# Mount static files for PDF streaming and frontend
try:
    from pathlib import Path
    static_dir = Path(settings.UPLOAD_DIR).parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
        logger.info(f"✅ Static files mounted from {static_dir}")
except Exception as e:
    logger.warning(f"Could not mount static files: {e}")

# Include routers in canonical order
app.include_router(health.router, prefix="/health", tags=["Health"])
app.include_router(sessions.router, prefix="/sessions", tags=["Sessions"])
app.include_router(upload.router, prefix="/upload", tags=["Upload"])
app.include_router(pdfs.router, prefix="/pdfs", tags=["PDF Streaming"])
app.include_router(extract_1a.router, prefix="/extract/1a", tags=["PDF Extraction"])
app.include_router(search.router, prefix="/search", tags=["Search"])
app.include_router(insights.router, prefix="/insights", tags=["Insights"])
app.include_router(persona.router, prefix="/persona", tags=["Persona Analysis"])
app.include_router(podcast.router, prefix="/podcast", tags=["Podcast"])
app.include_router(graph.router, prefix="/knowledge-graph", tags=["Knowledge Graph"])

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.API_TITLE,
        "version": settings.API_VERSION,
        "description": settings.API_DESCRIPTION,
        "docs_url": "/docs",
        "health_url": "/health",
        "status": "ready",
        "capabilities": {
            "pdf_extraction": True,
            "semantic_search": True,
            "keyword_search": True,
            "insights": bool(settings.GEMINI_API_KEY),
            "podcast_generation": bool(settings.GEMINI_API_KEY),
            "tts": bool(settings.AZURE_SPEECH_KEY or settings.GOOGLE_TTS_CREDENTIALS),
            "knowledge_graph": True
        }
    }


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Custom 404 handler."""
    return JSONResponse(
        status_code=404,
        content={
            "success": False,
            "error": "Not Found",
            "detail": f"The requested resource was not found: {request.url.path}",
            "status_code": 404
        }
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Custom 500 handler."""
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal Server Error",
            "detail": "An unexpected error occurred. Please try again later.",
            "status_code": 500
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )
