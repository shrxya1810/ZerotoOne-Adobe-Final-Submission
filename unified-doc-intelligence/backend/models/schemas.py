"""
Pydantic models and schemas for the unified document intelligence API.
"""
from datetime import datetime
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field
from enum import Enum


# Enums
class DocumentType(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"


class SearchType(str, Enum):
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


class TTSProvider(str, Enum):
    AZURE = "azure"
    GOOGLE = "google"
    LOCAL = "local"


# Base Models
class BaseResponse(BaseModel):
    """Base response model with common fields."""
    success: bool = True
    timestamp: datetime = Field(default_factory=datetime.now)
    message: Optional[str] = None


# Session Models
class SessionCreate(BaseModel):
    """Request model for creating a new session."""
    name: Optional[str] = Field(default=None, description="Optional session name")


class Session(BaseModel):
    """Session model."""
    session_id: str
    name: Optional[str] = None
    created_at: datetime
    last_accessed: datetime
    document_count: int = 0


class SessionResponse(BaseResponse):
    """Response model for session operations."""
    session: Session


# Document Models
class DocumentMetadata(BaseModel):
    """Document metadata model."""
    filename: str
    file_size: int
    upload_time: datetime
    page_count: Optional[int] = None
    title: Optional[str] = None
    author: Optional[str] = None
    doc_type: DocumentType


class DocumentInfo(BaseModel):
    """Document information with processing status."""
    filename: str
    status: str  # "uploaded", "processing", "indexed", "error"
    metadata: Optional[DocumentMetadata] = None
    error_message: Optional[str] = None


class UploadResponse(BaseResponse):
    """Response model for file uploads."""
    uploaded_files: List[DocumentInfo]
    session_id: str


# Search Models
class SearchRequest(BaseModel):
    """Request model for search operations."""
    query: str = Field(..., description="Search query")
    session_id: str = Field(..., description="Session ID to search within")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of results to return")
    search_type: SearchType = Field(default=SearchType.SEMANTIC, description="Type of search")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Optional filters")


class SearchResult(BaseModel):
    """Individual search result."""
    document: str = Field(..., description="Source document filename")
    content: str = Field(..., description="Matched content/snippet")
    page_number: int = Field(..., description="0-indexed page number")
    score: float = Field(..., description="Relevance score")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")


class SearchResponse(BaseResponse):
    """Response model for search operations."""
    query: str
    results: List[SearchResult]
    total_results: int
    search_type: SearchType


# Extract/Challenge 1a Models
class OutlineItem(BaseModel):
    """Outline item from PDF structure extraction."""
    level: int = Field(..., description="Heading level (1-6)")
    title: str = Field(..., description="Heading text")
    page_number: int = Field(..., description="0-indexed page number")
    bbox: Optional[List[float]] = Field(default=None, description="Bounding box coordinates [x0, y0, x1, y1]")


class ExtractResponse(BaseResponse):
    """Response model for PDF structure extraction (Challenge 1a)."""
    filename: str
    title: Optional[str] = Field(default=None, description="Extracted document title")
    outline: List[OutlineItem] = Field(..., description="Document structure/outline")
    page_count: int = Field(..., description="Total number of pages")
    processing_time: float = Field(..., description="Processing time in seconds")


# Insights Models
class InsightsRequest(BaseModel):
    """Request model for generating insights."""
    session_id: str = Field(..., description="Session ID")
    selected_text: str = Field(..., description="Selected text to analyze")
    top_k_related: int = Field(default=3, ge=1, le=10, description="Number of related chunks to include")
    focus_areas: Optional[List[str]] = Field(default=None, description="Specific areas to focus insights on")


class RelatedChunk(BaseModel):
    """Related content chunk."""
    document: str
    content: str
    page_number: int
    similarity_score: float


class SimplifiedInsights(BaseModel):
    """Simplified insights structure without redundancy."""
    summary: str = Field(..., description="Clean summary of selected text")
    sentiment_tone: str = Field(..., description="Sentiment/tone analysis")
    contradictions: str = Field(..., description="Contradictions if any")
    overlap_redundancy: str = Field(..., description="Overlap/redundancy if any")
    supporting_evidence: str = Field(..., description="Supporting evidence across all docs")
    external_context: str = Field(..., description="External contradictions, counterpoints, real-world examples, surprising insights, related articles")


class DocumentContextInsights(BaseModel):
    """Document context analysis insights."""
    summary: str = Field(..., description="Brief explanation of selected text")
    sentiment_tone: str = Field(..., description="Sentiment/tone analysis")
    contextual_summary: str = Field(..., description="How text fits in whole document")
    contradictions: str = Field(..., description="Internal contradictions")
    overlap_redundancy: str = Field(..., description="Overlap/redundancy check")
    supporting_evidence: str = Field(..., description="Supporting evidence from document")


class ExternalContextInsights(BaseModel):
    """External context analysis insights."""
    external_contradictions: str = Field(..., description="External source contradictions")
    counterpoints: str = Field(..., description="Opposite views in literature/research")
    real_world_examples: str = Field(..., description="Real-world examples")
    surprising_insights: str = Field(..., description="Unexpected insights from external context")
    related_articles: str = Field(..., description="Suggested readings/references")


class StructuredInsights(BaseModel):
    """Structured insights containing all analysis types."""
    document_context: DocumentContextInsights
    external_context: ExternalContextInsights
    overall_analysis: str = Field(..., description="Synthesis of internal and external analysis")


class InsightsResponse(BaseResponse):
    """Response model for insights generation."""
    selected_text: str
    insights: StructuredInsights = Field(..., description="Structured insights analysis")
    simplified_insights: Optional[SimplifiedInsights] = Field(None, description="Simplified non-redundant insights")
    related_chunks: List[RelatedChunk] = Field(..., description="Related content")
    sources: List[str] = Field(..., description="Source documents")


# Ask AI Models
class AskAIRequest(BaseModel):
    """Request model for Ask AI functionality."""
    question: str = Field(..., description="Question to ask")
    session_id: str = Field(..., description="Session ID for context")
    include_sources: bool = Field(default=True, description="Include source citations")
    max_context_chunks: int = Field(default=5, ge=1, le=10, description="Max context chunks to use")


class SourceCitation(BaseModel):
    """Source citation for AI responses."""
    document: str
    page_number: int
    snippet: str
    relevance_score: float


class AskAIResponse(BaseResponse):
    """Response model for Ask AI."""
    question: str
    answer: str
    sources: List[SourceCitation] = Field(default=[], description="Source citations")
    confidence: Optional[float] = Field(default=None, description="Confidence score")


# Persona Analysis Models
class PersonaAnalysisRequest(BaseModel):
    """Request model for persona analysis."""
    session_id: str = Field(..., description="Session ID")
    persona: str = Field(..., description="Persona/role (e.g., 'Travel Planner')")
    job_to_be_done: str = Field(..., description="Specific task to accomplish")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of top sections to return")


class PersonaSection(BaseModel):
    """Section relevant to persona analysis."""
    document: str
    section_title: str
    content: str
    page_number: int
    importance_rank: int
    relevance_score: float


class PersonaAnalysisResponse(BaseResponse):
    """Response model for persona analysis."""
    persona: str
    job_to_be_done: str
    extracted_sections: List[PersonaSection]
    summary: Optional[str] = Field(default=None, description="Cross-document summary")
    recommendations: Optional[List[str]] = Field(default=None, description="Actionable recommendations")
    relevant_sections: Optional[List[str]] = Field(default=None, description="Key relevant sections with page references")
    gaps: Optional[List[str]] = Field(default=None, description="Information gaps identified")
    evaluation_metrics: Optional[Dict] = Field(default=None, description="Evaluation metrics for the analysis")
    evaluation_report: Optional[str] = Field(default=None, description="Human-readable evaluation report")


# Podcast Models
class PodcastScriptRequest(BaseModel):
    """Request model for podcast script generation."""
    session_id: str = Field(..., description="Session ID for context")
    topic: str = Field(..., description="Main topic for the podcast")
    style: str = Field(default="conversational", description="Podcast style (conversational, educational, professional)")
    max_sections: int = Field(default=5, ge=1, le=10, description="Maximum sections to include")


class PodcastSegment(BaseModel):
    """Individual podcast segment/line."""
    speaker: str = Field(..., description="Speaker name")
    text: str = Field(..., description="Spoken content")
    duration_seconds: Optional[int] = Field(default=None, description="Segment duration in seconds")


class PodcastScriptResponse(BaseResponse):
    """Response model for podcast script generation."""
    topic: str
    style: str
    script: str = Field(..., description="Full script text")
    segments: List[Dict[str, Any]] = Field(default=[], description="Script segments")
    estimated_duration: str = Field(..., description="Estimated duration as MM:SS")


class PodcastGenerationRequest(BaseModel):
    """Request model for podcast audio generation."""
    session_id: str = Field(..., description="Session ID")
    script: Optional[str] = Field(default=None, description="Script text to convert to audio")
    topic: Optional[str] = Field(default=None, description="Topic to generate script for")
    voice: str = Field(default="en-US-AriaNeural", description="Voice to use for TTS")
    tts_provider: TTSProvider = Field(default=TTSProvider.AZURE, description="TTS provider")


class PodcastGenerationResponse(BaseResponse):
    """Response model for podcast audio generation."""
    session_id: str
    topic: str
    audio_file_path: str
    script_used: str
    voice_used: str
    provider_used: str
    estimated_duration: str


# Knowledge Graph Models
class GraphNode(BaseModel):
    """Knowledge graph node."""
    id: str
    label: str
    type: str  # "concept", "document", "entity"
    properties: Dict[str, Any] = Field(default={})


class GraphEdge(BaseModel):
    """Knowledge graph edge."""
    source: str
    target: str
    relationship: str
    weight: float = Field(default=1.0)
    properties: Dict[str, Any] = Field(default={})


class KnowledgeGraphResponse(BaseResponse):
    """Response model for knowledge graph."""
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    session_id: str
    graph_stats: Dict[str, Any] = Field(default={})


# Health Check Models
class HealthResponse(BaseResponse):
    """Health check response."""
    status: str = "healthy"
    version: str
    uptime: float
    services: Dict[str, str] = Field(default={}, description="Service status")
    capabilities: Dict[str, bool] = Field(default={}, description="Available capabilities")


# Error Models
class ErrorResponse(BaseModel):
    """Error response model."""
    success: bool = False
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    error_code: Optional[str] = None


# Generic response for listing
class ListResponse(BaseResponse):
    """Generic list response."""
    items: List[Any]
    total_count: int
    page: int = Field(default=1)
    per_page: int = Field(default=50)


# Document list response
class DocumentListResponse(BaseResponse):
    """Response for listing documents in a session."""
    session_id: str
    documents: List[DocumentInfo]
    total_count: int
