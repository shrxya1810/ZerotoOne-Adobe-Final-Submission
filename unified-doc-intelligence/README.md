# 🧠 Unified Document Intell## 📡 API Endpoints Referencegence

A modular monolith FastAPI application that consolidates document processing, search, analysis, and content generation capabilities from multiple repositories into a single, powerful backend.

## 🎯 Features

### Core Capabilities

- **📄 PDF Processing**: Extract text and metadata from PDF documents using Challenge 1a logic
- **🔍 Multi-Modal Search**: Semantic search (MPNet), keyword search (SQLite FTS5), and hybrid search
- **💡 AI Insights**: Generate insights and answer questions using Google Gemini AI
- **👤 Persona Analysis**: Cross-document analysis for specific personas and jobs-to-be-done
- **🎙️ Podcast Generation**: AI-generated scripts and multi-provider TTS audio generation
- **🕸️ Knowledge Graphs**: Visualize document relationships, concepts, and entities
- **💾 Session Management**: Organize documents and maintain context across interactions

### Technical Stack

- **Backend**: FastAPI 0.111.1 with uvicorn
- **Embeddings**: MPNet (multi-qa-mpnet-base-dot-v1) primary, MiniLM-L6-v2 fallback
- **Vector Search**: FAISS with cosine similarity
- **Full-Text Search**: SQLite FTS5 with Porter stemming
- **AI**: Google Gemini 1.5 Flash for insights and content generation
- **TTS**: Azure Speech Services, Google Cloud TTS, and local TTS support
- **PDF Processing**: PyMuPDF with Gemini AI enhancement
- **Storage**: SQLite for metadata, FAISS for vectors
- **Frontend**: Pure HTML/CSS/JavaScript with Adobe PDF Embed API

## � API Endpoints Reference

### Root Endpoint

#### GET `/`
**Purpose**: API information and capability detection  
**Returns**: Basic API metadata and available features  
**Use Case**: Initial frontend discovery of backend capabilities

---

### 🏥 Health & Monitoring (`/health`)

#### GET `/health/`
**Purpose**: Comprehensive system health check  
**Returns**: Detailed status of all services, uptime, and capabilities  
**Use Case**: Monitoring dashboards, debugging service issues

#### GET `/health/ping`
**Purpose**: Simple connectivity test  
**Returns**: Basic "pong" response  
**Use Case**: Load balancer health checks

#### GET `/health/ready`
**Purpose**: Readiness probe for container orchestration  
**Returns**: Service readiness status  
**Use Case**: Kubernetes readiness probes

#### GET `/health/live`
**Purpose**: Liveness probe for container orchestration  
**Returns**: Service liveness status  
**Use Case**: Kubernetes liveness probes

---

### 📄 PDF Extraction (`/extract/1a`)

#### POST `/extract/1a/process-pdf`
**Purpose**: Extract title and hierarchical outline from PDF  
**Input**: Multipart form with PDF file  
**Returns**: Document title, structured outline with page numbers (0-indexed)  
**Use Case**: Challenge 1a implementation, PDF navigation structure

#### POST `/extract/1a/process-pdf-legacy`
**Purpose**: Legacy PDF processing endpoint  
**Input**: PDF file upload  
**Returns**: Basic extraction results  
**Use Case**: Backwards compatibility

#### GET `/extract/1a/health`
**Purpose**: PDF extraction service health  
**Returns**: Service status  
**Use Case**: Service-specific monitoring

---

### 🗂️ Session Management (`/sessions`)

#### POST `/sessions/create`
**Purpose**: Create new document session  
**Input**: Optional session name  
**Returns**: Session ID and metadata  
**Use Case**: Start new document analysis workflow

#### GET `/sessions/{session_id}`
**Purpose**: Get session information  
**Input**: Session ID  
**Returns**: Session details, document count, creation time  
**Use Case**: Session status checking, frontend state restoration

#### DELETE `/sessions/{session_id}`
**Purpose**: Delete session and all associated data  
**Input**: Session ID  
**Returns**: Deletion confirmation  
**Use Case**: Cleanup, user-initiated deletion

#### POST `/sessions/{session_id}/extend`
**Purpose**: Extend session expiration time  
**Input**: Session ID, optional extension duration  
**Returns**: New expiration time  
**Use Case**: Keep active sessions alive

#### POST `/sessions/cleanup`
**Purpose**: Clean up expired sessions  
**Returns**: Number of sessions cleaned  
**Use Case**: Automated maintenance, storage optimization

---

### 📤 File Upload (`/upload`)

#### POST `/upload/batch`
**Purpose**: Upload and process multiple PDF documents  
**Input**: Session ID, multiple PDF files via multipart form  
**Returns**: Upload status, document IDs, processing status  
**Use Case**: Bulk document ingestion with background processing

#### GET `/upload/status/{session_id}`
**Purpose**: Check document processing status  
**Input**: Session ID  
**Returns**: Processing status for all documents in session  
**Use Case**: Progress tracking, UI updates during processing

---

### 📁 PDF Serving (`/pdfs`)

#### GET `/pdfs/{filename}`
**Purpose**: Serve PDF files with streaming support  
**Input**: PDF filename  
**Returns**: PDF file with range request support  
**Use Case**: Adobe PDF Embed API file serving

#### GET `/pdfs/{session_id}/{filename}`
**Purpose**: Serve session-specific PDF files  
**Input**: Session ID, filename  
**Returns**: PDF file with CORS headers  
**Use Case**: Session-scoped PDF access with security

#### GET `/pdfs/list/{session_id}`
**Purpose**: List all PDFs in a session  
**Input**: Session ID  
**Returns**: Array of PDF documents with metadata  
**Use Case**: Document browser, file selection interfaces

#### GET `/pdfs/`
**Purpose**: Global PDF listing  
**Returns**: All available PDF files  
**Use Case**: Admin interfaces, system overview

---

### 🔍 Search (`/search`)

#### POST `/search/semantic`
**Purpose**: Semantic vector-based search  
**Input**: Session ID, query text, optional filters  
**Returns**: Ranked results with similarity scores and source citations  
**Use Case**: Natural language search, concept-based discovery

#### POST `/search/keyword`
**Purpose**: Full-text keyword search  
**Input**: Session ID, query terms, optional filters  
**Returns**: Ranked results with relevance scores  
**Use Case**: Exact phrase matching, traditional search

#### POST `/search/hybrid`
**Purpose**: Combined semantic + keyword search  
**Input**: Session ID, query, weighting parameters  
**Returns**: Merged and re-ranked results from both methods  
**Use Case**: Best-of-both-worlds search experience

#### GET `/search/test/{session_id}`
**Purpose**: Test search functionality with sample queries  
**Input**: Session ID  
**Returns**: Search test results  
**Use Case**: Development, debugging search quality

---

### 💡 AI Insights (`/insights`)

#### POST `/insights/selection`
**Purpose**: Generate AI insights for selected text  
**Input**: Session ID, selected text, context  
**Returns**: AI-generated insights, implications, recommendations  
**Use Case**: Text analysis, contextual understanding

#### POST `/insights/ask`
**Purpose**: Ask questions about documents using AI  
**Input**: Session ID, question, optional context  
**Returns**: AI answer with source citations  
**Use Case**: Document Q&A, research assistance

#### GET `/insights/test/{session_id}`
**Purpose**: Test insights functionality  
**Input**: Session ID  
**Returns**: Sample insights  
**Use Case**: Development, feature validation

---

### 👤 Persona Analysis (`/persona`)

#### POST `/persona/analyze`
**Purpose**: Cross-document analysis for specific persona and job-to-be-done  
**Input**: Session ID, persona description, job-to-be-done  
**Returns**: Persona-focused analysis, recommendations, action items  
**Use Case**: Targeted content analysis, role-specific insights

#### GET `/persona/personas`
**Purpose**: Get available persona templates  
**Returns**: Predefined persona types and descriptions  
**Use Case**: Persona selection interfaces

#### GET `/persona/test/{session_id}`
**Purpose**: Test persona analysis  
**Input**: Session ID  
**Returns**: Sample persona analysis  
**Use Case**: Feature demonstration

---

### 🎙️ Podcast Generation (`/podcast`)

#### POST `/podcast/generate_script`
**Purpose**: Generate podcast script from documents  
**Input**: Session ID, topic, style preferences  
**Returns**: Structured podcast script with speaker segments  
**Use Case**: Content repurposing, audio content creation

#### POST `/podcast/generate_audio`
**Purpose**: Convert script to audio using TTS  
**Input**: Session ID, script, voice preferences  
**Returns**: Audio file URL and metadata  
**Use Case**: Podcast production, accessibility

#### GET `/podcast/audio/{session_id}/{filename}`
**Purpose**: Serve generated podcast audio files  
**Input**: Session ID, audio filename  
**Returns**: Audio file stream  
**Use Case**: Audio playback in frontend

#### GET `/podcast/voices`
**Purpose**: List available TTS voices  
**Returns**: Voice options with languages and providers  
**Use Case**: Voice selection interfaces

#### GET `/podcast/history/{session_id}`
**Purpose**: Get podcast generation history  
**Input**: Session ID  
**Returns**: Previous podcasts and scripts  
**Use Case**: Content management, re-generation

#### DELETE `/podcast/audio/{session_id}/{filename}`
**Purpose**: Delete generated audio files  
**Input**: Session ID, filename  
**Returns**: Deletion confirmation  
**Use Case**: Storage cleanup

---

### 🕸️ Knowledge Graph (`/knowledge-graph`)

#### GET `/knowledge-graph/generate/{session_id}`
**Purpose**: Generate knowledge graph from session documents  
**Input**: Session ID  
**Returns**: Graph nodes, edges, and visualization data  
**Use Case**: Document relationship visualization

#### GET `/knowledge-graph/node/{session_id}/{node_id}`
**Purpose**: Get detailed information about a graph node  
**Input**: Session ID, node ID  
**Returns**: Node details, connections, related content  
**Use Case**: Interactive graph exploration

#### GET `/knowledge-graph/search/{session_id}`
**Purpose**: Search within knowledge graph  
**Input**: Session ID, search query  
**Returns**: Matching nodes and relationships  
**Use Case**: Graph-based content discovery

#### GET `/knowledge-graph/stats/{session_id}`
**Purpose**: Get knowledge graph statistics  
**Input**: Session ID  
**Returns**: Node counts, relationship types, coverage metrics  
**Use Case**: Graph analysis, quality assessment

---

## �🚀 Quick Start

### 1. Environment Setup

```bash
# Clone or navigate to the project
cd unified-doc-intelligence

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
```

### 2. Configure Environment Variables

Edit `.env` file with your API keys:

```bash
# Required for AI features
GEMINI_API_KEY=your_gemini_api_key_here

# Required for TTS features
AZURE_SPEECH_KEY=your_azure_speech_key
AZURE_SPEECH_REGION=your_azure_region

# Optional: Google Cloud TTS
GOOGLE_TTS_CREDENTIALS=path/to/google/credentials.json

# Optional: Custom settings
UPLOAD_DIR=./storage/uploads
SQLITE_DB_PATH=./storage/database.db
LOG_LEVEL=INFO
```

### 3. Start the Backend

```bash
# Method 1: Using the startup script
python start_backend.py

# Method 2: Direct uvicorn command
uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
```

### 4. Start the Frontend

```bash
# In a new terminal
python serve_frontend.py
```

### 5. Access the Application

- **Frontend**: http://localhost:3000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 📱 Using the Application

### Step 1: Create a Session

1. Enter a session name (e.g., "Research Project")
2. Click "Create Session"

### Step 2: Upload Documents

1. Drag and drop PDF files or click to browse
2. Click "Upload Selected"
3. Wait for processing to complete

### Step 3: Search and Analyze

- **Search Tab**: Perform semantic, keyword, or hybrid searches
- **Analysis Tab**: Analyze documents from specific persona perspectives
- **Podcast Tab**: Generate audio summaries and scripts

## 🔧 Quick API Reference

### Core Endpoints

- `POST /sessions/create` - Create a new session
- `POST /upload/batch` - Upload PDF documents
- `GET /pdfs/list/{session_id}` - List session documents

### Search Endpoints

- `POST /search/semantic` - Semantic similarity search
- `POST /search/keyword` - Full-text keyword search
- `POST /search/hybrid` - Combined semantic + keyword search

### Analysis Endpoints

- `POST /insights/selection` - Generate insights for selected text
- `POST /insights/ask` - Ask questions about documents
- `POST /persona/analyze` - Persona-based cross-document analysis

### Content Generation

- `POST /podcast/generate_script` - Generate podcast scripts
- `POST /podcast/generate_audio` - Generate podcast audio
- `GET /knowledge-graph/generate/{session_id}` - Generate knowledge graphs

### PDF Processing

- `POST /extract/1a/process-pdf` - Challenge 1a PDF extraction (0-indexed pages)
- `GET /pdfs/{session_id}/{filename}` - Stream PDF files

## 🏗️ Architecture

### Project Structure

```
unified-doc-intelligence/
├── backend/
│   ├── app.py                 # Main FastAPI application
│   ├── settings.py            # Configuration management
│   ├── models/
│   │   └── schemas.py         # Pydantic models
│   ├── services/
│   │   ├── embed.py          # Embedding service (MPNet/MiniLM)
│   │   ├── storage.py        # SQLite storage management
│   │   ├── index.py          # FAISS vector indexing
│   │   ├── pdf_extract.py    # PDF processing
│   │   └── tts.py            # Text-to-speech services
│   └── routers/
│       ├── health.py         # Health checks
│       ├── sessions.py       # Session management
│       ├── upload.py         # File upload handling
│       ├── search.py         # Search endpoints
│       ├── insights.py       # AI insights
│       ├── persona.py        # Persona analysis
│       ├── podcast.py        # Podcast generation
│       └── graph.py          # Knowledge graphs
├── frontend/
│   └── index.html            # Web interface
├── requirements.txt          # Python dependencies
├── .env.example             # Environment template
├── start_backend.py         # Backend startup script
└── serve_frontend.py        # Frontend development server
```

### Service Architecture

- **Embedding Service**: Global caching, model hot-swapping (MPNet → MiniLM)
- **Storage Service**: SQLite with FTS5, session isolation, document metadata
- **Index Service**: Per-session FAISS indices, file persistence
- **TTS Service**: Multi-provider support with voice mapping
- **PDF Extract Service**: Challenge 1a logic with Gemini enhancement

## 🔑 Key Features in Detail

### Multi-Modal Search

- **Semantic Search**: Uses MPNet embeddings with FAISS for similarity matching
- **Keyword Search**: SQLite FTS5 with Porter stemming for exact term matching
- **Hybrid Search**: Combines and re-ranks semantic + keyword results

### Persona Analysis

- Cross-document analysis for specific user personas
- Job-to-be-done focused content extraction
- AI-powered insights and recommendations
- Relevance ranking and importance scoring

### Podcast Generation

- AI-generated scripts with multiple styles (conversational, educational, professional)
- Multi-provider TTS support (Azure, Google, Local)
- Voice selection and audio file management
- Integration with document content for context

### Knowledge Graphs

- Entity and concept extraction from documents
- Document similarity analysis with semantic embeddings
- Node and edge relationship mapping
- Graph statistics and search capabilities

## ⚙️ Configuration

### Environment Variables

| Variable                 | Description                           | Required | Default                      |
| ------------------------ | ------------------------------------- | -------- | ---------------------------- |
| `GEMINI_API_KEY`         | Google Gemini API key for AI features | Yes      | -                            |
| `AZURE_SPEECH_KEY`       | Azure Speech Services key             | For TTS  | -                            |
| `AZURE_SPEECH_REGION`    | Azure region                          | For TTS  | -                            |
| `GOOGLE_TTS_CREDENTIALS` | Google Cloud TTS credentials          | Optional | -                            |
| `UPLOAD_DIR`             | Directory for uploaded files          | No       | `./storage/uploads`          |
| `SQLITE_DB_PATH`         | SQLite database path                  | No       | `./storage/database.db`      |
| `EMBEDDING_MODEL`        | Primary embedding model               | No       | `multi-qa-mpnet-base-dot-v1` |
| `FALLBACK_MODEL`         | Fallback embedding model              | No       | `all-MiniLM-L6-v2`           |
| `LOG_LEVEL`              | Logging level                         | No       | `INFO`                       |
| `DEBUG`                  | Enable debug mode                     | No       | `false`                      |

### Model Configuration

- **Primary Embedding Model**: `sentence-transformers/multi-qa-mpnet-base-dot-v1`
- **Fallback Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **AI Model**: `gemini-1.5-flash` for content generation
- **Chunk Size**: 500 characters with 50-character overlap

## 🛠️ Development

### Adding New Features

1. Define Pydantic schemas in `backend/models/schemas.py`
2. Implement business logic in `backend/services/`
3. Create API endpoints in `backend/routers/`
4. Register routers in `backend/app.py`
5. Update frontend JavaScript for new functionality

### Testing

```bash
# Test individual components
python -m py_compile backend/app.py

# Test API endpoints
curl http://localhost:8000/health

# Run with different configurations
DEBUG=true LOG_LEVEL=DEBUG python start_backend.py
```

### Deployment

- Configure production environment variables
- Use gunicorn or similar WSGI server for production
- Set up reverse proxy (nginx) for static file serving
- Configure SSL/TLS certificates for HTTPS

## 📚 API Documentation

Once the backend is running, visit:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

## 🔍 Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: Ensure all dependencies are installed: `pip install -r requirements.txt`
2. **Embedding Model Loading**: First run may take time to download models
3. **CORS Issues**: Backend runs on 8000, frontend on 3000 - CORS is pre-configured
4. **Memory Usage**: MPNet models require ~2GB RAM - will fallback to MiniLM if needed
5. **API Key Issues**: Check `.env` file and ensure API keys are valid

### Performance Tips

- Use environment variables to limit embedding cache size
- Regularly clear storage directories to manage disk space
- Monitor memory usage with larger document collections
- Use smaller chunk sizes for faster processing

## 🤝 Contributing

This project consolidates functionality from multiple repositories:

- Challenge 1a: PDF extraction and hierarchy enhancement
- BackendwithPodcast: TTS and podcast generation (persona + KG removed)
- unified_doc_backend: Knowledge graphs and insights
- pdf_chat_mpnet: Embedding and search capabilities

## 📄 License

This project combines multiple codebases. Please check individual repository licenses for specific components.

## 🆘 Support

For issues and questions:

1. Check the troubleshooting section above
2. Review API documentation at `/docs`
3. Check server logs for detailed error messages
4. Verify environment variable configuration
