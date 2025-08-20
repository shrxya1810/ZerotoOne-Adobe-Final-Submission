// Add a global ref for AdobeEmbedViewer to allow LeftSidebar to access it
if (typeof window !== "undefined") {
  (window as any).__adobeRef = null;
}
// Outline/Headings type for extracted PDF structure
export interface OutlineHeading {
  id?: string;
  text: string;
  level: number;
  page?: number;
  children?: OutlineHeading[];
}

export type OutlineType = OutlineHeading[];
import { create } from "zustand";
import { devtools } from "zustand/middleware";
import { hybridSearch, semanticSearch, keywordSearch } from "../../shared/api";

export interface RelatedSuggestion {
  id: string;
  text: string;
  documentName: string;
  confidence: number;
  page?: number;
}

export interface SearchResult {
  id: string;
  document: string;
  content: string;
  page_number: number;
  score: number;
  metadata?: Record<string, any>;
}

export interface DocumentConfiguration {
  documentId: string;
  persona: string;
  prompt: string;
  appliedAt: Date;
}
export interface Document {
  id: string;
  name: string;
  type: "pdf" | "docx" | "txt" | "web";
  content?: string;
  url?: string;
  uploadedAt: Date;
  size: number;
  isCurrentReading?: boolean;
}

export interface TextSelection {
  text: string;
  startOffset: number;
  endOffset: number;
  documentId: string;
  context?: string;
}

export interface SimplifiedInsights {
  summary: string;
  sentiment_tone: string;
  contradictions: string;
  overlap_redundancy: string;
  supporting_evidence: string;
  external_context: string;
}

export interface DocumentContextInsights {
  summary: string;
  sentiment_tone: string;
  contextual_summary: string;
  contradictions: string;
  overlap_redundancy: string;
  supporting_evidence: string;
}

export interface ExternalContextInsights {
  external_contradictions: string;
  counterpoints: string;
  real_world_examples: string;
  surprising_insights: string;
  related_articles: string;
}

export interface StructuredInsights {
  document_context: DocumentContextInsights;
  external_context: ExternalContextInsights;
  overall_analysis: string;
}

export interface AIInsight {
  id: string;
  type: "summary" | "related" | "contradiction" | "enhancement";
  content: string;
  confidence: number;
  sourceDocuments: string[];
  relatedSections: Array<{
    documentId: string;
    text: string;
    page?: number;
  }>;
  // Add structured insights for new format
  structured_insights?: StructuredInsights;
  simplified_insights?: SimplifiedInsights;
}

export interface KnowledgeGraphNode {
  id: string;
  label: string;
  type: "concept" | "entity" | "topic";
  documentIds: string[];
  connections: string[];
  position?: { x: number; y: number };
}

export interface AppState {
  // Journey Bar Progress
  currentStep: 0 | 1 | 2 | 3 | 4;
  // Documents
  documents: Document[];
  currentDocument: Document | null;
  isUploading: boolean;

  // Outline/Headings per document
  outlines: Record<string, OutlineType>;
  // Active outline section per document
  activeOutlineSection: Record<string, string>;

  // Text Selection & AI Features
  currentSelection: TextSelection | null;
  aiInsights: AIInsight[];
  isGeneratingInsights: boolean;

  // UI State
  leftSidebarOpen: boolean;
  rightSidebarOpen: boolean;
  currentView: "workspace" | "knowledge-graph" | "upload";
  searchQuery: string;
  searchScope: "current" | "all" | "selection";
  searchResults: SearchResult[];
  isSearching: boolean;
  showSearchDropdown: boolean;

  // Homepage State Management
  homepageState: "hero" | "upload" | "features";
  scrollProgress: number;
  isTransitioning: boolean;
  hasScrolledPastHero: boolean;

  // Knowledge Graph
  knowledgeGraph: KnowledgeGraphNode[];
  selectedGraphNode: KnowledgeGraphNode | null;

  // Audio
  isGeneratingAudio: boolean;
  currentAudioUrl: string | null;

  // Document Configuration
  documentConfigurations: Record<string, DocumentConfiguration>;

  // Session
  sessionId: string | null;
  sessionName: string;
  isDirty: boolean;

  relatedSuggestions: RelatedSuggestion[];
  setRelatedSuggestions: (suggestions: RelatedSuggestion[]) => void;
}

export interface AppActions {
  // Journey Bar actions
  setCurrentStep: (step: 0 | 1 | 2 | 3 | 4) => void;
  // Document actions
  addDocuments: (documents: Document[]) => void;
  removeDocument: (id: string) => void;
  setCurrentDocument: (document: Document | null) => void;
  setIsUploading: (loading: boolean) => void;

  // Outline actions
  setOutline: (docId: string, outline: OutlineType) => void;
  setActiveOutlineSection: (docId: string, sectionId: string) => void;

  // Selection & AI actions
  setCurrentSelection: (selection: TextSelection | null) => void;
  setAIInsights: (insights: AIInsight[]) => void;
  setIsGeneratingInsights: (generating: boolean) => void;
  addAIInsight: (insight: AIInsight) => void;

  // UI actions
  toggleLeftSidebar: () => void;
  toggleRightSidebar: () => void;
  setCurrentView: (view: AppState["currentView"]) => void;
  setSearchQuery: (query: string) => void;
  setSearchScope: (scope: AppState["searchScope"]) => void;
  setSearchResults: (results: SearchResult[]) => void;
  setIsSearching: (searching: boolean) => void;
  setShowSearchDropdown: (show: boolean) => void;
  performSearch: (query: string, sessionId: string) => Promise<void>;

  // Homepage state actions
  setHomepageState: (state: AppState["homepageState"]) => void;
  setScrollProgress: (progress: number) => void;
  setIsTransitioning: (transitioning: boolean) => void;
  setHasScrolledPastHero: (scrolled: boolean) => void;

  // Knowledge Graph actions
  setKnowledgeGraph: (nodes: KnowledgeGraphNode[]) => void;
  setSelectedGraphNode: (node: KnowledgeGraphNode | null) => void;

  // Audio actions
  setIsGeneratingAudio: (generating: boolean) => void;
  setCurrentAudioUrl: (url: string | null) => void;

  // Session actions
  setSessionId: (id: string | null) => void;
  setSessionName: (name: string) => void;
  setIsDirty: (dirty: boolean) => void;
  resetSession: () => void;

  // Document Configuration actions
  setDocumentConfiguration: (config: DocumentConfiguration) => void;
  getDocumentConfiguration: (
    documentId: string,
  ) => DocumentConfiguration | null;
}

const initialState: AppState = {
  currentStep: 0,
  documents: [],
  currentDocument: null,
  isUploading: false,

  outlines: {},
  activeOutlineSection: {},

  currentSelection: null,
  aiInsights: [],
  isGeneratingInsights: false,

  leftSidebarOpen: true,
  rightSidebarOpen: true,
  currentView: "upload",
  searchQuery: "",
  searchScope: "all",
  searchResults: [],
  isSearching: false,
  showSearchDropdown: false,

  homepageState: "hero",
  scrollProgress: 0,
  isTransitioning: false,
  hasScrolledPastHero: false,

  knowledgeGraph: [],
  selectedGraphNode: null,

  isGeneratingAudio: false,
  currentAudioUrl: null,

  // Document Configuration
  documentConfigurations: {},

  sessionId: null,
  sessionName: "",
  isDirty: false,
  relatedSuggestions: [],
  setRelatedSuggestions: () => {},
};

export const useAppStore = create<AppState & AppActions>()(
  devtools(
    (set, get) => ({
      ...initialState,
      // Outline actions
      setOutline: (docId, outline) =>
        set((state) => ({
          outlines: { ...state.outlines, [docId]: outline },
          isDirty: true,
        })),
      setActiveOutlineSection: (docId, sectionId) =>
        set((state) => ({
          activeOutlineSection: { ...state.activeOutlineSection, [docId]: sectionId },
        })),
      // Journey Bar actions
      setCurrentStep: (step) => set({ currentStep: step }),

      // Document actions
      addDocuments: (documents) =>
        set((state) => ({
          documents: [...state.documents, ...documents],
          isDirty: true,
        })),

      removeDocument: (id) =>
        set((state) => ({
          documents: state.documents.filter((doc) => doc.id !== id),
          currentDocument:
            state.currentDocument?.id === id ? null : state.currentDocument,
          isDirty: true,
        })),

      setCurrentDocument: (document) =>
        set({
          currentDocument: document,
          currentView: document ? "workspace" : "upload",
        }),

      setIsUploading: (isUploading) => set({ isUploading }),

      // Selection & AI actions
      setCurrentSelection: (selection) => set({ currentSelection: selection }),

      setAIInsights: (insights) => set({ aiInsights: insights }),

      setIsGeneratingInsights: (generating) =>
        set({ isGeneratingInsights: generating }),

      addAIInsight: (insight) =>
        set((state) => ({ aiInsights: [...state.aiInsights, insight] })),

      // UI actions
      toggleLeftSidebar: () =>
        set((state) => ({ leftSidebarOpen: !state.leftSidebarOpen })),

      toggleRightSidebar: () =>
        set((state) => ({ rightSidebarOpen: !state.rightSidebarOpen })),

      setCurrentView: (view) => set({ currentView: view }),

      setSearchQuery: (query) => set({ searchQuery: query }),

      setSearchScope: (scope) => set({ searchScope: scope }),

      setSearchResults: (results) => set({ searchResults: results }),

      setIsSearching: (searching) => set({ isSearching: searching }),

      setShowSearchDropdown: (show) => set({ showSearchDropdown: show }),

      performSearch: async (query, sessionId) => {
        if (!query.trim() || !sessionId) {
          console.log("Search skipped: empty query or session ID", {
            query,
            sessionId,
          });
          return;
        }

        const { setSearchResults, setIsSearching } = useAppStore.getState();

        try {
          setIsSearching(true);
          console.log("Performing search:", { query, sessionId });

          // Try hybrid search first, fallback to semantic if it fails
          let response;
          let searchMethod = "hybrid";

          try {
            console.log("🔍 Attempting hybrid search...");
            response = await hybridSearch(sessionId, query, 10);
            console.log("✅ Hybrid search successful");
          } catch (hybridError) {
            console.warn(
              "❌ Hybrid search failed, falling back to semantic search:",
              hybridError,
            );
            searchMethod = "semantic";
            try {
              console.log("🔍 Attempting semantic search...");
              response = await semanticSearch(sessionId, query, 10);
              console.log("✅ Semantic search successful");
            } catch (semanticError) {
              console.warn(
                "❌ Semantic search failed, falling back to keyword search:",
                semanticError,
              );
              searchMethod = "keyword";
              console.log("🔍 Attempting keyword search...");
              response = await keywordSearch(sessionId, query, 10);
              console.log("✅ Keyword search successful");
            }
          }

          // Transform response to SearchResult format
          const searchResults: SearchResult[] =
            response.results?.map((result: any, index: number) => ({
              id: `${result.document}_${result.page_number}_${index}`,
              document: result.document,
              content: result.content,
              page_number: result.page_number,
              score: result.score,
              metadata: {
                ...result.metadata,
                searchMethod, // Add search method to metadata
              },
            })) || [];

          console.log(
            `🎯 Search completed using ${searchMethod} method:`,
            searchResults,
          );
          setSearchResults(searchResults);
        } catch (error) {
          console.error("❌ All search methods failed:", error);
          setSearchResults([]);
        } finally {
          setIsSearching(false);
        }
      },

      // Homepage state actions
      setHomepageState: (state) => set({ homepageState: state }),

      setScrollProgress: (progress) => set({ scrollProgress: progress }),

      setIsTransitioning: (transitioning) =>
        set({ isTransitioning: transitioning }),

      setHasScrolledPastHero: (scrolled) =>
        set({ hasScrolledPastHero: scrolled }),

      // Knowledge Graph actions
      setKnowledgeGraph: (nodes) => set({ knowledgeGraph: nodes }),

      setSelectedGraphNode: (node) => set({ selectedGraphNode: node }),

      // Audio actions
      setIsGeneratingAudio: (generating) =>
        set({ isGeneratingAudio: generating }),

      setCurrentAudioUrl: (url) => set({ currentAudioUrl: url }),

      // Session actions
      setSessionId: (id) => set({ sessionId: id }),

      setSessionName: (name) => set({ sessionName: name, isDirty: true }),

      setIsDirty: (dirty) => set({ isDirty: dirty }),

      resetSession: () => set(initialState),

      // Document Configuration actions
      setDocumentConfiguration: (config) =>
        set((state) => ({
          documentConfigurations: {
            ...state.documentConfigurations,
            [config.documentId]: config,
          },
          isDirty: true,
        })),

      getDocumentConfiguration: (documentId) => {
        const state = get();
        return state.documentConfigurations[documentId] || null;
      },

      relatedSuggestions: [],
      setRelatedSuggestions: (suggestions) =>
        set({ relatedSuggestions: suggestions }),
    }),
    {
      name: "research-canvas-store",
    },
  ),
);
