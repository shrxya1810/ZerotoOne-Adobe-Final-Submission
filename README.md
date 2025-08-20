# 🚀 Connect-the-Dots: Intelligent PDF Analyser

## 🔑 **IMPORTANT: Adobe Embed API Key**
```
ADOBE_EMBED_API_KEY: 98e7a97c303a4803b955a5af21f1185f
```
**This API key is required for PDF viewing functionality. Set it as `ADOBE_EMBED_API_KEY` while running docker.**

## 🐳 **Quick Start with Docker**

```bash
# Build the Docker image
docker build -t intelligentpdf .

# Run with all environment variables
docker run \
  -e ADOBE_EMBED_API_KEY=98e7a97c303a4803b955a5af21f1185f \
  -e GEMINI_API_KEY=your_gemini_api_key \
  -e AZURE_TTS_KEY=your_azure_tts_key \
  -e AZURE_TTS_ENDPOINT=your_azure_endpoint \
  -p 8080:8080 \
  -p 8000:8000 \
  intelligentpdf
```

**Access the application at:**
- APP: http://localhost:8080 \
Wait for backend to start. We are good to go when **INFO:     Application startup complete** is displayed

---

## 🎥 **Video Demo & Walkthrough**

<div align="center">

[![Connect-the-Dots Demo](https://img.shields.io/badge/🎬-Watch%20Demo%20Video-blue?style=for-the-badge&logo=youtube)](https://drive.google.com/file/d/1RcAaZyuTcGVVVDMv-PSx8c70OKbpVHP-/view?usp=sharing)

**📺 [Click here to watch the full demo](https://drive.google.com/file/d/1RcAaZyuTcGVVVDMv-PSx8c70OKbpVHP-/view?usp=sharing)**

*Experience Connect-the-Dots in action: See how AI transforms static PDFs into dynamic knowledge networks*

</div>

---

## 🎯 **Project Overview**

**Connect-the-Dots** is a revolutionary research platform that transforms passive document reading into an active, engaging, and insightful knowledge-building journey. Built for researchers, students, analysts, developers, and writers, it addresses the core challenges of modern research workflows:

### **🚨 The Problem We Solve**
- **Passive Consumption**: Static PDFs lead to shallow understanding and weak retention
- **Information Fragmentation**: Lost connections between concepts across multiple documents
- **Time-Intensive Analysis**: Manual extraction of insights and contradictions is slow
- **Context Limitations**: Traditional reading misses cross-document insights and relationships

### **💡 Our Solution**
Connect-the-Dots leverages **AI-powered document analysis**, **persona-based research perspectives**, and **interconnected knowledge graphs** to transform scattered reading into connected understanding. Users discover hidden insights, connect ideas across sources, and build comprehensive knowledge networks.

---

## ✨ **Core Features & Capabilities**

### 🔬 **Smart Document Processing & Navigation**
- **Bulk PDF Upload**: Upload multiple documents simultaneously with intelligent workspace management
- **Adobe Embed API Integration**: Professional PDF viewer with advanced navigation capabilities
- **Smart Outline Extraction**: AI-powered document structure analysis with interactive navigation
- **Go-to-Page Functionality**: Direct navigation from outline items and search results
- **Document Mini-Map**: Visual representation of content density and insights distribution

### 🎭 **Advanced Persona-Based Research Analysis**
- **6 Specialized Research Personas**:
  - **🔬 Researcher**: Academic focus with methodology analysis, hypothesis generation, literature review
  - **🎓 Student**: Learning optimization with concept explanations, study guides, knowledge gaps identification
  - **📊 Analyst**: Business intelligence with data insights, trend analysis, strategic recommendations
  - **💻 Developer**: Technical documentation focus with code analysis, architecture insights, implementation guidance
  - **✍️ Writer**: Content creation with fact-checking, narrative structure, and style analysis
  - **🎯 Custom**: User-defined analysis with personalized prompts and custom objectives
- **Intelligent Prompt Engineering**: Context-aware prompts based on document type and content
- **Cross-Document Analysis**: Maintains persona consistency across multiple documents
- **Performance Tracking**: Analytics on which personas generate the most valuable insights

### 🤖 **AI-Powered Intelligence Engine**
- **Google Gemini 2.5 Flash Integration**: Advanced LLM for document analysis and insight generation
- **Multi-Type Insights Generation**:
  - **Smart Summaries**: Context-aware document overviews with key takeaways
  - **Related Content Discovery**: Cross-document connections and thematic relationships
  - **Contradiction Detection**: AI-powered identification of conflicting information
  - **Enhancement Suggestions**: Recommendations for content improvement and expansion
- **Confidence Scoring**: AI-generated confidence levels for each insight with source traceability
- **Hybrid Search Capabilities**: Combines semantic (FAISS) and keyword (SQLite FTS5) search for optimal results

### 🎧 **Professional Audio & Podcast Generation**
- **Azure Cognitive Services TTS**: Premium neural voices with natural speech patterns for podcast features
- **AI-Generated Podcast Scripts**: Transform technical content into engaging conversational dialogues
- **Multi-Voice Support**: Multiple voice options (AriaNeural, JennyNeural, etc.) with SSML support
- **Advanced Audio Controls**: Playback speed (0.5x-2x), volume control, progress seeking
- **Audio Export**: Download generated audio in MP3/WAV formats for offline consumption
- **Real-Time Processing**: Streaming synthesis with chunk-based processing for large documents

### 🧠 **Dynamic Knowledge Graph Visualization**
- **Mathematical Graph Algorithms**:
  - **Cosine Similarity**: Vectorized similarity calculation using sklearn for optimal performance
  - **Jaccard Similarity**: Text-based similarity with batch processing optimization
  - **Length Similarity Penalty**: Document length normalization for fair comparison
  - **Community Detection**: Advanced clustering algorithms for concept grouping
- **Interactive Exploration**: Click and explore knowledge nodes with real-time analytics
- **Cross-Document Linking**: Semantic relationships with similarity scoring and confidence metrics
- **Graph Analytics**: Centrality metrics, clustering coefficients, modularity scores

### 🔍 **Advanced Search & Discovery**
- **Semantic Search**: FAISS-based similarity search with multi-qa-mpnet-base-dot-v1 embeddings
- **Keyword Search**: SQLite FTS5 with BM25 ranking for precise phrase matching
- **Hybrid Fusion**: Intelligent combination of semantic and keyword results with configurable weights
- **Result Diversification**: Advanced merging algorithms for diverse, relevant results
- **Optimized Chunking**: Intelligent content chunking with metadata preservation

### 🎨 **Modern, Responsive User Interface**
- **Glassmorphism Design**: Beautiful, modern UI with glass-like effects and smooth animations
- **Dark Theme**: Professional dark theme optimized for extended research sessions
- **Responsive Layout**: Fully responsive design for all device sizes with mobile-first approach
- **GSAP Animations**: Professional-grade animations with timeline control and performance optimization
- **Intuitive Navigation**: Clear, logical navigation structure with collapsible sidebar panels

---

## 🏗️ **System Architecture & Technology Stack**

### **Frontend Architecture**
- **React 18**: Latest React features with concurrent rendering, Suspense, and automatic batching
- **TypeScript 5.0+**: Full type safety with advanced type inference and enhanced developer experience
- **Vite 5.0+**: Lightning-fast build tool with HMR, native ESM, and optimized bundling
- **Zustand**: Lightweight, scalable state management with TypeScript support and DevTools integration

### **Backend & AI Services**
- **FastAPI**: High-performance Python API with automatic OpenAPI documentation
- **Google Gemini 2.5 Flash**: Advanced LLM for document analysis and insight generation
- **Azure Cognitive Services**: Premium text-to-speech with neural voice synthesis for podcast features
- **FAISS Vector Database**: Efficient similarity search with GPU acceleration support
- **SQLite Database**: Lightweight, embedded database for session and metadata storage

### **Document Processing Pipeline**
- **PyMuPDF (fitz)**: High-performance PDF text extraction and analysis
- **Sentence Transformers**: Multi-language sentence embeddings with fallback support
- **Spacy NLP**: Advanced natural language processing with entity recognition
- **Transformers**: Hugging Face model integration for text analysis

### **UI/UX & Visualization**
- **Tailwind CSS 3.0+**: Utility-first CSS with JIT compilation and custom design system
- **Shadcn/ui**: Radix-based accessible components with Tailwind integration
- **React Force Graph**: Interactive graph visualization with WebGL acceleration
- **D3.js**: Data-driven graph computations and force simulations
- **GSAP**: Professional-grade animations with timeline control and performance optimization

---

## 🚀 **Getting Started**

### **Prerequisites**
- Node.js 18.0+ 
- npm or yarn package manager
- Modern web browser with ES6+ support
- Docker (for containerized deployment)

### **Local Development Setup**

```bash
# Clone the repository
git clone https://github.com/yourusername/connect-the-dots.git
cd connect-the-dots

# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Run type checking
npm run typecheck

# Run tests
npm run test
```

### **Environment Configuration**

#### **Required API Keys**
```bash
# Copy environment template
cp .env.example .env

# CRITICAL: Set Adobe Embed API Key for PDF viewing
export VITE_ADOBE_CLIENT_ID="98e7a97c303a4803b955a5af21f1185f"

# Configure additional API keys
export GEMINI_API_KEY="your_gemini_api_key"
export AZURE_TTS_KEY="your_azure_speech_key"
export AZURE_SPEECH_REGION="eastus"
```

#### **Docker Deployment**
```bash
# Build the Docker image
docker build -t intelligentpdf .

# Run with all environment variables
docker run \
  -e ADOBE_EMBED_API_KEY=98e7a97c303a4803b955a5af21f1185f \
  -e GEMINI_API_KEY=your_gemini_api_key \
  -e AZURE_TTS_KEY=your_azure_tts_key \
  -e AZURE_TTS_ENDPOINT=your_azure_endpoint \
  -p 8080:8080 \
  -p 8000:8000 \
  intelligentpdf
```

#### **Environment Variables Reference**
| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `ADOBE_EMBED_API_KEY` | ✅ **YES** | Adobe Embed API Key: `98e7a97c303a4803b955a5af21f1185f` | - |
| `GEMINI_API_KEY` | ✅ **YES** | Google Gemini API for AI analysis | - |
| `AZURE_TTS_KEY` | ⚡ Optional | Azure Speech Services for audio generation | - |
| `AZURE_TTS_ENDPOINT` | ⚡ Optional | Custom Azure TTS endpoint | Default Azure endpoint |
| `AZURE_SPEECH_REGION` | ⚡ Optional | Azure region | `eastus` |

---

## 📁 **Project Structure**

```
ZerotoOne-Adobe-Final-Submission/
├── Connect-the-Dots-clean-main/           # Main project directory
│   ├── Connect-the-Dots-clean-main/       # Nested project structure
│   │   ├── client/                        # Frontend React application
│   │   │   ├── components/                # Reusable UI components
│   │   │   │   ├── ui/                    # Shadcn/ui components
│   │   │   │   ├── workspace/             # Workspace-specific components
│   │   │   │   ├── persona/               # Persona analysis components
│   │   │   │   └── AdobeEmbedViewer.tsx   # Adobe PDF viewer integration
│   │   │   ├── pages/                     # Application pages and routing
│   │   │   ├── hooks/                     # Custom React hooks
│   │   │   ├── lib/                       # Utility libraries and Zustand store
│   │   │   └── global.css                 # Global styles and CSS variables
│   │   ├── server/                        # Backend API (Python/FastAPI)
│   │   │   ├── app.py                     # Main FastAPI application
│   │   │   ├── pdf_extractor.py           # PDF processing and extraction
│   │   │   ├── hierarchy_enhancer.py      # Document structure enhancement
│   │   │   └── requirements.txt           # Python dependencies
│   │   ├── shared/                        # Shared utilities and types
│   │   ├── netlify/                       # Netlify serverless functions
│   │   └── package.json                   # Frontend dependencies
├── unified-doc-intelligence/              # Alternative backend implementation
│   ├── backend/                           # FastAPI backend
│   │   ├── app.py                         # Main application with router mounting
│   │   ├── routers/                       # API endpoint definitions
│   │   │   ├── search.py                  # Semantic, keyword, and hybrid search
│   │   │   ├── insights.py                # AI-powered insights generation
│   │   │   ├── persona.py                 # Persona-based analysis
│   │   │   ├── podcast.py                 # Audio generation and TTS
│   │   │   ├── graph.py                   # Knowledge graph algorithms
│   │   │   └── extract_1a.py             # PDF structure extraction
│   │   ├── services/                      # Business logic services
│   │   ├── models/                        # Data schemas and models
│   │   └── settings.py                    # Configuration management
│   ├── start_backend.py                   # Backend startup script
│   └── requirements.txt                   # Python dependencies
└── Dockerfile                             # Docker configuration
```

---

## 🔧 **Configuration & Customization**

### **Persona Configuration**
The system supports 6 predefined research personas, each optimized for different use cases:

- **Researcher**: Academic and professional research focus with methodology analysis
- **Student**: Learning and academic project optimization with concept explanations
- **Analyst**: Business intelligence and data-driven insights with trend analysis
- **Developer**: Technical documentation and code analysis with implementation guidance
- **Writer**: Content creation and fact-checking support with narrative structure analysis
- **Custom**: User-defined analysis requirements with personalized prompts

### **AI Insight Types**
- **Summary**: Concise document overviews with key takeaways
- **Related Content**: Cross-document connections and thematic relationships
- **Contradictions**: AI-powered identification of conflicting information
- **Enhancements**: Suggestions for content improvement and expansion

---

## 🌐 **API Integration & Endpoints**

### **Core API Endpoints**
```typescript
// Document Processing
POST /extract/1a/process-pdf          # Smart outline extraction
POST /upload                           # Document upload and indexing

// Search & Discovery
POST /search/semantic                  # Semantic search with FAISS
POST /search/keyword                   # Keyword search with SQLite FTS5
POST /search/hybrid                    # Hybrid search fusion

// AI Analysis
POST /insights/selection               # Context-specific insights
POST /persona_analysis                 # Persona-based analysis
POST /ask_ai                           # Global question answering

// Audio Generation
POST /podcast/script                   # AI-generated podcast scripts
POST /podcast/audio                    # Text-to-speech audio generation

// Knowledge Graph
GET /knowledge-graph                   # Dynamic graph generation
```

### **Backend Requirements**
The frontend integrates with backend services providing:

- **Document Processing**: PDF parsing, text extraction, and structure analysis
- **AI Analysis**: Gemini 2.5 Flash integration for intelligent insights
- **Knowledge Graph**: Graph database for relationship storage and visualization
- **Audio Generation**: Azure TTS and podcast script generation
- **User Management**: Session management and user preferences

---

## 📊 **Performance & Advanced Optimizations**

### **Frontend Performance**
- **Code Splitting**: Route-based and component-level splitting for optimal loading
- **Lazy Loading**: Dynamic component imports with React.lazy() and Suspense
- **Bundle Optimization**: Tree-shaking with Rollup optimization and bundle analysis
- **Virtual Scrolling**: Efficient rendering of large document lists
- **Memoization**: React.memo() and useMemo() for expensive calculations

### **AI & Backend Optimizations**
- **Gemini API Optimization**:
  - Token management with 4000 input / 500 output limits
  - Batch processing with 5 simultaneous requests
  - Response caching with 7-day TTL
  - Request debouncing with 300ms delay
- **Content Processing**:
  - Intelligent chunking with metadata preservation
  - Quality assessment before processing
  - Multi-threaded document analysis
  - Streaming processing for large documents

### **Performance Metrics**
- **Lighthouse Score**: 95+ performance rating
- **Core Web Vitals**: FCP < 1.5s, LCP < 2.5s, CLS < 0.1, FID < 100ms
- **AI Response Time**: < 3s for insights generation
- **Graph Rendering**: < 500ms for 1000+ nodes
- **Document Loading**: < 2s for 10MB PDFs

---

## 🧪 **Testing & Quality Assurance**

### **Testing Strategy**
- **Unit Tests**: Component and utility function testing with Vitest
- **Integration Tests**: API integration testing
- **E2E Tests**: User workflow testing with Playwright
- **Accessibility Tests**: Screen reader and keyboard navigation compliance

### **Code Quality**
- **ESLint**: JavaScript/TypeScript linting with custom rules
- **Prettier**: Code formatting and consistency
- **TypeScript**: Static type checking and type safety
- **Husky**: Git hooks for pre-commit quality gates

---

## 🚀 **Deployment & Production**

### **Production Build**
```bash
# Build optimized production bundle
npm run build

# Preview production build
npm run preview

# Deploy to Netlify
netlify deploy --prod
```

### **Docker Production**
```bash
# Multi-stage production build
docker build -t connect-the-dots:prod .

# Run production container
docker run -d \
  --name connect-the-dots \
  -p 80:8080 \
  -p 8000:8000 \
  --restart unless-stopped \
  connect-the-dots:prod
```

---

## 🤝 **Contributing & Development**

We welcome contributions from the community! Please read our contributing guidelines:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Commit your changes** (`git commit -m 'Add amazing feature'`)
4. **Push to the branch** (`git push origin feature/amazing-feature`)
5. **Open a Pull Request**

### **Development Guidelines**
- Follow TypeScript best practices and maintain type safety
- Maintain consistent code style with Prettier
- Write comprehensive tests for new features
- Update documentation for API changes
- Follow accessibility guidelines (WCAG 2.1 AA)

---

## 📈 **Roadmap & Future Plans**

### **Phase 1: Core Platform** ✅
- [x] Document upload and workspace management
- [x] AI-powered insights generation
- [x] Persona-based research configuration
- [x] Knowledge graph visualization
- [x] Adobe Embed API integration
- [x] Audio generation and podcast features

### **Phase 2: Advanced Features** 🚧
- [ ] Real-time collaboration and shared workspaces
- [ ] Advanced knowledge graph algorithms and analytics
- [ ] Multi-language support for global accessibility
- [ ] Mobile application (React Native)
- [ ] Advanced search filters and saved searches

### **Phase 3: Enterprise Features** 📋
- [ ] Team workspaces and role-based access control
- [ ] Advanced analytics and usage insights
- [ ] API marketplace for third-party integrations
- [ ] Enterprise integrations (Notion, Slack, Google Drive, Microsoft Teams)
- [ ] Advanced security and compliance features

---

## 📚 **Documentation & Resources**

- **API Reference**: [docs/api.md](docs/api.md)
- **Component Library**: [docs/components.md](docs/components.md)
- **Deployment Guide**: [docs/deployment.md](docs/deployment.md)
- **Contributing Guide**: [docs/contributing.md](docs/contributing.md)
- **User Guide**: [docs/user-guide.md](docs/user-guide.md)

---

## 🏆 **Acknowledgments & Technologies**

- **AI Integration**: Powered by Google Gemini 2.5 Flash
- **PDF Viewing**: Adobe Embed API for professional document rendering
- **UI Components**: Built with Shadcn/ui and Tailwind CSS
- **Animations**: Enhanced with GSAP for smooth interactions
- **Icons**: Beautiful icons from Lucide React
- **Audio Generation**: Azure Cognitive Services for premium TTS and podcast features
- **Backend Framework**: FastAPI for high-performance API development

---

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📞 **Support & Contact**

- **Documentation**: [docs.connectthedots.ai](https://docs.connectthedots.ai)
- **Issues**: [GitHub Issues](https://github.com/yourusername/connect-the-dots/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/connect-the-dots/issues)
- **Email**: support@connectthedots.ai
- **Demo Video**: [Watch Full Demo](https://drive.google.com/file/d/1RcAaZyuTcGVVVDMv-PSx8c70OKbpVHP-/view?usp=sharing)

---

<div align="center">

**Made with ❤️ by Team ZeroToOne**

[![GitHub stars](https://img.shields.io/github/stars/yourusername/connect-the-dots?style=social)](https://github.com/yourusername/connect-the-dots)
[![GitHub forks](https://img.shields.io/github/forks/yourusername/connect-the-dots?style=social)](https://github.com/yourusername/connect-the-dots)
[![GitHub issues](https://img.shields.io/badge/GitHub%20Issues-0-brightgreen)](https://github.com/yourusername/connect-the-dots/issues)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Connect-the-Dots: Transforming Passive Reading into Active Knowledge Building** 🚀

</div>

