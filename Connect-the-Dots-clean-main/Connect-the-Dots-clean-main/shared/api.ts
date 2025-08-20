// Upload a PDF and get extracted headings from backend
export async function processPdf(file: File) {
  const formData = new FormData();
  formData.append("file", file);
  const response = await fetch("/api/extract/1a/process-pdf", {
    method: "POST",
    body: formData,
  });
  if (!response.ok) {
    const errorResult = await response.json();
    throw new Error(errorResult.detail || "An unknown error occurred");
  }
  const result = await response.json();
  // Extract the outline array from the response
  if (result.success && Array.isArray(result.outline)) {
    // Map backend field names to frontend expected field names
    const mappedOutline = result.outline.map((item) => ({
      id: item.id,
      text: item.title, // Map 'title' to 'text'
      level: item.level,
      page: item.page_number, // Map 'page_number' to 'page'
      children: item.children,
    }));
    return mappedOutline;
  } else {
    throw new Error("Failed to extract outline from PDF");
  }
}
/**
 * Shared code between client and server
 * Useful to share types between client and server
 * and/or small pure JS functions that can be used on both client and server
 */

/**
 * Example response type for /api/demo
 */
export interface DemoResponse {
  message: string;
}

// Create a new session
export async function createSession(name?: string) {
  const response = await fetch("/api/sessions/create", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
  if (!response.ok) throw new Error("Failed to create session");
  return response.json();
}

// Upload documents to a session
export async function uploadDocuments(sessionId: string, files: File[]) {
  const formData = new FormData();
  formData.append("session_id", sessionId);
  files.forEach((file) => formData.append("files", file));

  const response = await fetch("/api/upload/batch", {
    method: "POST",
    body: formData,
  });
  if (!response.ok) throw new Error("Failed to upload documents");
  return response.json();
}

// Get upload status for a session
export async function getUploadStatus(sessionId: string) {
  const response = await fetch(`/api/upload/status/${sessionId}`);
  if (!response.ok) throw new Error("Failed to get upload status");
  return response.json();
}

// Get semantic search results
export async function semanticSearch(
  sessionId: string,
  query: string,
  topK: number = 5,
) {
  console.log("🧠 Making semantic search request:", { sessionId, query, topK });

  const requestBody = { query, session_id: sessionId, top_k: topK };
  console.log("📤 Request body:", requestBody);

  const response = await fetch("/api/search/semantic", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(requestBody),
  });

  console.log(
    "📡 Semantic search response status:",
    response.status,
    response.statusText,
  );

  if (!response.ok) {
    const errorText = await response.text();
    console.error("❌ Semantic search error response:", {
      status: response.status,
      statusText: response.statusText,
      error: errorText,
    });
    throw new Error(
      `Failed to perform semantic search: ${response.status} - ${errorText}`,
    );
  }

  const result = await response.json();
  console.log("✅ Semantic search result:", result);

  // Check if results contain actual content
  if (result.results && result.results.length > 0) {
    console.log("📝 Sample content from first result:", {
      document: result.results[0].document,
      content: result.results[0].content?.substring(0, 100) + "...",
      hasPageMarker: result.results[0].content?.includes("--- Page"),
    });
  }

  return result;
}

// Get keyword search results
export async function keywordSearch(
  sessionId: string,
  query: string,
  topK: number = 5,
) {
  console.log("🔍 Making keyword search request:", { sessionId, query, topK });

  const requestBody = { query, session_id: sessionId, top_k: topK };
  console.log("📤 Request body:", requestBody);

  const response = await fetch("/api/search/keyword", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(requestBody),
  });

  console.log(
    "📡 Keyword search response status:",
    response.status,
    response.statusText,
  );

  if (!response.ok) {
    const errorText = await response.text();
    console.error("❌ Keyword search error response:", {
      status: response.status,
      statusText: response.statusText,
      error: errorText,
    });
    throw new Error(
      `Failed to perform keyword search: ${response.status} - ${errorText}`,
    );
  }

  const result = await response.json();
  console.log("✅ Keyword search result:", result);

  // Check if results contain actual content
  if (result.results && result.results.length > 0) {
    console.log("📝 Sample content from first result:", {
      document: result.results[0].document,
      content: result.results[0].content?.substring(0, 100) + "...",
      hasPageMarker: result.results[0].content?.includes("--- Page"),
    });
  }

  return result;
}

// Get hybrid search results (semantic + keyword)
export async function hybridSearch(
  sessionId: string,
  query: string,
  topK: number = 5,
) {
  const response = await fetch("/api/search/hybrid", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, session_id: sessionId, top_k: topK }),
  });
  if (!response.ok) throw new Error("Failed to perform hybrid search");
  return response.json();
}

// Get AI insights for selected text
export async function getAIInsights(
  sessionId: string,
  selectedText: string,
  context?: string,
) {
  const response = await fetch("/api/insights/selection", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: sessionId,
      selected_text: selectedText,
      context,
    }),
  });
  if (!response.ok) throw new Error("Failed to get AI insights");
  return response.json();
}

// Get related suggestions for a text selection (updated to use semantic search)
export async function getRelatedSuggestions(
  sessionId: string,
  selection: string,
) {
  return semanticSearch(sessionId, selection, 5);
}

// Generate podcast script from selected text or topic
export async function generatePodcastScript(
  sessionId: string,
  topic: string,
  style: string = "conversational",
) {
  const response = await fetch("/api/podcast/generate_script", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: sessionId,
      topic,
      style,
      max_sections: 5,
    }),
  });
  if (!response.ok) throw new Error("Failed to generate podcast script");
  return response.json();
}

// Generate audio from script or topic
export async function generatePodcastAudio(
  sessionId: string,
  options: {
    script?: string;
    topic?: string;
    voice?: string;
    tts_provider?: string;
  },
) {
  const response = await fetch("/api/podcast/generate_audio", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: sessionId,
      ...options,
    }),
  });
  if (!response.ok) throw new Error("Failed to generate podcast audio");
  return response.json();
}

// Get available TTS voices
export async function getAvailableVoices() {
  const response = await fetch("/api/podcast/voices");
  if (!response.ok) throw new Error("Failed to get available voices");
  return response.json();
}

// Get podcast generation history
export async function getPodcastHistory(sessionId: string) {
  const response = await fetch(`/api/podcast/history/${sessionId}`);
  if (!response.ok) throw new Error("Failed to get podcast history");
  return response.json();
}

// Persona Analysis API
export async function analyzePersona(
  sessionId: string,
  persona: string,
  jobToBeDone: string,
  topK: number = 5,
) {
  const response = await fetch("/api/persona/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: sessionId,
      persona,
      job_to_be_done: jobToBeDone,
      top_k: topK,
    }),
  });
  if (!response.ok) throw new Error("Failed to analyze persona");
  return response.json();
}

// Get persona suggestions
export async function getPersonaSuggestions() {
  const response = await fetch("/api/persona/personas");
  if (!response.ok) throw new Error("Failed to get persona suggestions");
  return response.json();
}

// Ask AI functionality for chat
export async function askAI(
  sessionId: string,
  question: string,
  includeSource: boolean = true,
) {
  const response = await fetch("/api/insights/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: sessionId,
      question,
      include_sources: includeSource,
      max_context_chunks: 5,
    }),
  });
  if (!response.ok) throw new Error("Failed to ask AI");
  return response.json();
}

// Knowledge Graph API Functions

export interface GraphNode {
  id: string;
  label: string;
  type: "document";
  properties: Record<string, any>;
}

export interface GraphEdge {
  source: string;
  target: string;
  relationship: string;
  weight: number;
  properties?: Record<string, any>;
}

export interface KnowledgeGraphResponse {
  nodes: GraphNode[];
  edges: GraphEdge[];
  session_id: string;
  graph_stats: {
    total_nodes: number;
    total_edges: number;
    document_nodes?: number;
    concept_nodes?: number;
    entity_nodes?: number;
    centrality_metrics?: Record<string, any>;
    communities?: Record<string, number>;
    communities_count?: number;
  };
  message: string;
}

// Generate knowledge graph for a session
export async function generateKnowledgeGraph(
  sessionId: string,
  similarityThreshold: number = 0.7,
): Promise<KnowledgeGraphResponse> {
  const response = await fetch(
    `/api/knowledge-graph/generate/${sessionId}?similarity_threshold=${similarityThreshold}`,
  );
  if (!response.ok) throw new Error("Failed to generate knowledge graph");
  return response.json();
}

// Get detailed information about a specific node
export async function getNodeDetails(sessionId: string, nodeId: string) {
  const response = await fetch(
    `/api/knowledge-graph/node/${sessionId}/${nodeId}`,
  );
  if (!response.ok) throw new Error("Failed to get node details");
  return response.json();
}

// Search for nodes in the knowledge graph
export async function searchGraphNodes(
  sessionId: string,
  query: string,
  nodeTypes: string = "document,concept,entity",
) {
  const response = await fetch(
    `/api/knowledge-graph/search/${sessionId}?query=${encodeURIComponent(query)}&node_types=${nodeTypes}`,
  );
  if (!response.ok) throw new Error("Failed to search graph nodes");
  return response.json();
}

// Get graph statistics
export async function getGraphStatistics(sessionId: string) {
  const response = await fetch(`/api/knowledge-graph/stats/${sessionId}`);
  if (!response.ok) throw new Error("Failed to get graph statistics");
  return response.json();
}

// Get relationship details between two nodes
export async function getRelationshipDetails(
  sessionId: string,
  sourceId: string,
  targetId: string,
) {
  const response = await fetch(
    `/api/knowledge-graph/relationship/${sessionId}?source_id=${encodeURIComponent(sourceId)}&target_id=${encodeURIComponent(targetId)}`,
  );
  if (!response.ok) throw new Error("Failed to get relationship details");
  return response.json();
}
