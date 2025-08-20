"""
Settings and configuration for the unified document intelligence backend.
"""
import os
from typing import Optional, List
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Adobe Settings
    ADOBE_EMBED_API_KEY: Optional[str] = Field(default=None, description="Adobe Embed API key")
    
    # API Settings
    API_TITLE: str = "Unified Document Intelligence API"
    API_VERSION: str = "1.0.0"
    API_DESCRIPTION: str = "Unified backend for PDF processing, search, insights, and podcast generation"
    
    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # CORS Settings
    CORS_ORIGINS: List[str] = ["*"]
    
    # Embedding Model Settings
    EMBEDDING_MODEL: str = Field(default="multi-qa-mpnet-base-dot-v1", 
                                description="Default: MPNet, fallback: all-MiniLM-L6-v2")
    EMBEDDING_CACHE_SIZE: int = 1000  # Max cached embeddings
    
    # Storage Settings
    UPLOAD_DIR: str = Field(default="./data/uploads", description="Directory for uploaded files")
    FAISS_DIR: str = Field(default="./data/faiss_data", description="Directory for FAISS indices")
    SQLITE_DB: str = Field(default="./data/db.sqlite", description="SQLite database path")
    TEMP_DIR: str = Field(default="./data/temp", description="Temporary files directory")
    
    # Search Settings
    DEFAULT_TOP_K: int = 5
    MAX_TOP_K: int = 50
    CHUNK_SIZE: int = 2000  # 2KB chunks - balanced between performance and granularity
    CHUNK_OVERLAP: int = 200
    
    # AI/LLM Settings
    LLM_PROVIDER: str = Field(default="gemini", description="LLM provider")
    GOOGLE_APPLICATION_CREDENTIALS: Optional[str] = Field(default=None, description="Google Application Credentials")
    GEMINI_API_KEY: Optional[str] = Field(default=None, description="Google Gemini API key")
    GEMINI_MODEL: str = "gemini-2.5-flash"
    GEMINI_MAX_TOKENS: int = 8192
    GEMINI_TEMPERATURE: float = 0.1
    
    # Gemini Optimization Settings
    GEMINI_BATCH_SIZE: int = 5
    GEMINI_CACHE_TTL: int = 3600 * 24 * 7  # 1 week
    GEMINI_MAX_RETRIES: int = 3
    GEMINI_RETRY_DELAY: float = 1.0
    GEMINI_MIN_REQUEST_INTERVAL: float = 0.1
    GEMINI_ENABLE_CACHING: bool = True
    GEMINI_ENABLE_BATCHING: bool = True
    
    # Token Optimization
    GEMINI_MAX_INPUT_TOKENS: int = 4000  # Reduced from 8192
    GEMINI_MAX_OUTPUT_TOKENS: int = 500   # Reduced from 8192
    
    # TTS Settings
    TTS_PROVIDER: str = Field(default="azure", description="TTS provider: azure, google, local")
    AZURE_TTS_KEY: Optional[str] = Field(default=None, description="Azure TTS key")
    AZURE_TTS_ENDPOINT: Optional[str] = Field(default=None, description="Azure TTS endpoint URL")
    # Legacy variables for backward compatibility
    AZURE_SPEECH_KEY: Optional[str] = Field(default=None, description="Azure Speech Services key (legacy)")
    AZURE_SPEECH_REGION: Optional[str] = Field(default=None, description="Azure Speech Services region (legacy)")
    GOOGLE_TTS_CREDENTIALS: Optional[str] = Field(default=None, description="Google TTS credentials JSON path")
    
    # Audio Settings
    AUDIO_SAMPLE_RATE: int = 24000
    AUDIO_FORMAT: str = "mp3"
    
    # Session Settings
    SESSION_TIMEOUT_HOURS: int = 24
    MAX_FILES_PER_SESSION: int = 50
    MAX_FILE_SIZE_MB: int = 100
    
    # PDF Processing Settings
    PDF_EXTRACT_IMAGES: bool = False
    PDF_EXTRACT_TABLES: bool = True
    PDF_MIN_TEXT_LENGTH: int = 50
    
    # Performance Settings
    MAX_CONCURRENT_UPLOADS: int = 5
    INDEX_BATCH_SIZE: int = 100
    CACHE_TTL_SECONDS: int = 3600
    
    # Graph Performance Settings
    SIMILARITY_CALCULATION_MODE: str = "optimized"  # "optimized", "standard", "threshold"
    SIMILARITY_MIN_THRESHOLD: float = 0.1  # Skip very dissimilar documents
    ENABLE_SIMILARITY_CACHING: bool = True
    MAX_DOCUMENTS_FOR_FULL_GRAPH: int = 50  # Use threshold mode above this
    
    # Relaxed Similarity Thresholds
    DEFAULT_SIMILARITY_THRESHOLD: float = 0.4  # Default user threshold
    STRONG_SIMILARITY_THRESHOLD: float = 0.8   # 0.8+ = strong similarity
    MODERATE_SIMILARITY_THRESHOLD: float = 0.6  # 0.6+ = moderate similarity
    WEAK_SIMILARITY_THRESHOLD: float = 0.4     # 0.4+ = weak but related
    
    # PDF Extraction Performance Settings
    MAX_BATCH_PDF_EXTRACTION: int = 10  # Maximum PDFs to process in one batch
    
    # Content Selection Settings
    CONTENT_SELECTION_STRATEGY: str = "auto"  # "auto", "diverse", "semantic", "structural", "hybrid"
    MAX_CONTENT_SELECTION_LENGTH: int = 4000  # Maximum content length for similarity
    TARGET_CHUNK_COUNT: int = 5  # Target number of chunks to select
    ENABLE_CONTENT_QUALITY_ASSESSMENT: bool = True
    
    # Debug Settings
    DEBUG: bool = Field(default=False, description="Enable debug mode")
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = [".env", "../.env", "../../.env"]  # Multiple .env file locations
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"  # Ignore extra fields that might be in .env


# Load environment variables from .env file if it exists
def load_environment():
    """Load environment variables from .env file with fallback."""
    import os
    print(f"Current working directory: {os.getcwd()}")
    
    # Try to find .env file in multiple locations
    env_paths = [
        Path(".env"),  # Current directory
        Path(__file__).parent / ".env",  # Same directory as settings.py (backend/.env)
        Path(__file__).parent.parent / ".env",  # Parent directory (unified-doc-intelligence/.env)
        Path(__file__).parent.parent.parent / ".env",  # Project root
    ]
    
    print("Checking for .env files in these locations:")
    for env_path in env_paths:
        abs_path = env_path.absolute()
        exists = env_path.exists()
        print(f"  {abs_path} - {'✅ EXISTS' if exists else '❌ NOT FOUND'}")
        
        if exists:
            print(f"✅ Loading environment from: {abs_path}")
            result = load_dotenv(abs_path, override=False)  # Don't override existing env vars
            print(f"load_dotenv result: {result}")
            
            # Test if variables were loaded
            test_vars = ['ADOBE_EMBED_API_KEY', 'GEMINI_API_KEY', 'AZURE_TTS_KEY']
            print("Testing loaded variables:")
            for var in test_vars:
                value = os.getenv(var)
                print(f"  {var}: {'✅ SET' if value else '❌ NOT SET'}")
            
            break
    else:
        print("❌ No .env file found in any location")

# Load environment before creating settings instance
load_environment()

# Global settings instance
settings = Settings()

# Log important environment variable status
def log_environment_status():
    """Log the status of important environment variables."""
    important_vars = {
        'ADOBE_EMBED_API_KEY': settings.ADOBE_EMBED_API_KEY,
        'GEMINI_API_KEY': settings.GEMINI_API_KEY,
        'AZURE_TTS_KEY': settings.AZURE_TTS_KEY,
        'AZURE_TTS_ENDPOINT': settings.AZURE_TTS_ENDPOINT,
    }
    
    print("=== Environment Variables Status ===")
    for var_name, var_value in important_vars.items():
        status = "✅ Set" if var_value else "❌ Not Set"
        print(f"{var_name}: {status}")
    print("===================================")

# Call this function when settings are loaded
log_environment_status()

# Ensure directories exist
def ensure_directories():
    """Create necessary directories if they don't exist."""
    directories = [
        settings.UPLOAD_DIR,
        settings.FAISS_DIR,
        settings.TEMP_DIR,
        os.path.dirname(settings.SQLITE_DB)
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

# Call on module import
ensure_directories()
