#!/usr/bin/env python3
"""
Start the unified document intelligence backend server.
"""
import uvicorn
import os
import sys
import logging
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set environment variables for development
os.environ.setdefault("LOG_LEVEL", "INFO")
os.environ.setdefault("DEBUG", "true")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_api_configuration():
    """Check and log API configuration status."""
    logger.info("=== Backend API Configuration Check ===")
    
    # Check Adobe API
    adobe_key = os.getenv("ADOBE_EMBED_API_KEY")
    logger.info(f"Adobe Embed API: {'✅ Present' if adobe_key else '❌ Missing'}")
    
    # Check Gemini API
    gemini_key = os.getenv("GEMINI_API_KEY")
    logger.info(f"Gemini API: {'✅ Present' if gemini_key else '❌ Missing'}")
    
    # Check Azure TTS
    azure_key = os.getenv("AZURE_TTS_KEY") or os.getenv("AZURE_SPEECH_KEY")
    azure_endpoint = os.getenv("AZURE_TTS_ENDPOINT")
    logger.info(f"Azure TTS Key: {'✅ Present' if azure_key else '❌ Missing'}")
    logger.info(f"Azure TTS Endpoint: {'✅ Present' if azure_endpoint else '❌ Missing'}")
    
    logger.info("========================================")


def check_and_setup_environment():
    """Check and setup environment files."""
    env_file = project_root / ".env"
    env_example = project_root / ".env.example"
    
    if not env_file.exists() and env_example.exists():
        logger.info("Creating .env file from .env.example...")
        with open(env_example, 'r') as src, open(env_file, 'w') as dst:
            dst.write(src.read())
        logger.info("✅ .env file created")
    
    return True


def test_critical_imports():
    """Test critical imports to catch issues early."""
    try:
        logger.info("Testing critical imports...")
        
        # Test settings
        from backend.settings import settings
        logger.info("✅ Settings imported")
        
        # Test schemas
        from backend.models.schemas import BaseResponse
        logger.info("✅ Schemas imported")
        
        # Test services (with graceful failure)
        try:
            from backend.services.embed import embedding_service
            logger.info("✅ Embedding service imported")
        except Exception as e:
            logger.warning(f"⚠️  Embedding service issue: {e}")
        
        try:
            from backend.services.storage import storage_service
            logger.info("✅ Storage service imported")
        except Exception as e:
            logger.warning(f"⚠️  Storage service issue: {e}")
        
        try:
            from backend.services.tts import tts_service
            logger.info("✅ TTS service imported")
        except Exception as e:
            logger.warning(f"⚠️  TTS service issue: {e}")
        
        # Test app
        from backend.app import app
        logger.info("✅ FastAPI app imported")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Critical import failed: {e}")
        return False


if __name__ == "__main__":
    print("🚀 Starting Unified Document Intelligence Backend")
    print("📍 Project root:", project_root)
    print("🔧 Environment: Development")
    print("📝 API Documentation: http://localhost:8000/docs")
    print("🏥 Health Check: http://localhost:8000/health")
    print()
    
    # Setup environment
    check_and_setup_environment()
    
    # Check API configuration
    check_api_configuration()
    
    # Test imports
    if not test_critical_imports():
        logger.error("❌ Critical imports failed, server may not start properly")
        sys.exit(1)
    
    logger.info("✅ All critical components loaded successfully")
    print()
    
    try:
        uvicorn.run(
            "backend.app:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info",
            access_log=True
        )
    except KeyboardInterrupt:
        print("\n👋 Backend server stopped")
    except Exception as e:
        print(f"❌ Server error: {e}")
        sys.exit(1)
