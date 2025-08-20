import { useRef, useEffect } from "react";
import { AdobeEmbedViewer, AdobeEmbedViewerHandle } from "../AdobeEmbedViewer";
import { useAppStore } from "../../lib/store";
import { FileText } from "lucide-react";
import { cn } from "../../lib/utils";
import gsap from "gsap";
import { getRelatedSuggestions } from "../../../shared/api";

// Add timeout type declaration
declare global {
  interface Window {
    __insightTimeout: number;
  }
}

interface DocumentViewerProps {
  className?: string;
}

export default function DocumentViewer({ className }: DocumentViewerProps) {
  const adobeRef = useRef<AdobeEmbedViewerHandle>(null);

  // Expose adobeRef globally for LeftSidebar access
  useEffect(() => {
    if (typeof window !== "undefined") {
      (window as any).__adobeRef = adobeRef;
    }
  }, [adobeRef]);

  // Listen for custom go-to-page events (for outline navigation)
  useEffect(() => {
    function handleGoToPageEvent(e: CustomEvent) {
      console.log("go-to-page event received:", e.detail);
      if (adobeRef.current && typeof e.detail?.page === "number") {
        const targetPage = e.detail.page;
        const searchText = e.detail?.text;

        // Validate page number is within reasonable bounds
        if (targetPage < 1 || targetPage > 200) {
          // Reasonable upper limit
          console.warn(
            `Invalid page number: ${targetPage}. Must be between 1 and 200.`,
          );
          return;
        }

        console.log(`Calling navigation with page ${targetPage} and text "${searchText}"`);
        
        // Use the new method that supports text highlighting
        if (searchText && adobeRef.current.gotoPageAndHighlight) {
          adobeRef.current.gotoPageAndHighlight(targetPage, searchText);
        } else {
          adobeRef.current.gotoPage(targetPage);
        }
      } else {
        console.warn(
          "Adobe ref not available or invalid page number:",
          e.detail,
        );
      }
    }
    window.addEventListener("go-to-page", handleGoToPageEvent as EventListener);
    return () => {
      window.removeEventListener(
        "go-to-page",
        handleGoToPageEvent as EventListener,
      );
    };
  }, []);

  const {
    currentDocument,
    sessionId,
    setCurrentSelection,
    currentSelection,
    setIsGeneratingInsights,
    setAIInsights,
    addAIInsight,
    setCurrentStep,
    setRelatedSuggestions,
  } = useAppStore();
  // Debug: track currentStep in this component
  const debugStep = useAppStore((state) => state.currentStep);
  useEffect(() => {
    console.log("[JourneyBar Debug] currentStep in DocumentViewer:", debugStep);
  }, [debugStep]);

  // Remove mock PDF state, use AdobeEmbedViewer instead

  useEffect(() => {
    if (currentDocument) {
      // Animate document load
      if (document.body) {
        gsap.fromTo(
          document.body,
          { opacity: 0, y: 20 },
          { opacity: 1, y: 0, duration: 0.5, ease: "power2.out" },
        );
      }
    }
  }, [currentDocument]);

  // Handle selection from AdobeEmbedViewer
  const handleAdobeSelection = async (text: string, pageNum?: number) => {
    if (text && currentDocument && text.trim().length > 10) {
      setCurrentSelection({
        text,
        startOffset: 0,
        endOffset: text.length,
        documentId: currentDocument.id,
        context: text,
      });
      setCurrentStep(2);

      // Debounce AI insights to prevent flooding
      clearTimeout((window as any).__insightTimeout);
      (window as any).__insightTimeout = setTimeout(() => {
        triggerAIInsights(text);
      }, 1000); // Wait 1 second after selection stops

      // Fetch related suggestions immediately (no debounce needed for search)
      if (sessionId) {
        try {
          const relatedResults = await getRelatedSuggestions(sessionId, text);
          const mappedSuggestions =
            relatedResults.results?.map((result) => ({
              id: `${result.document}_${result.page_number}`,
              text: result.content,
              documentName: result.document,
              confidence: result.score,
              page: result.page_number,
            })) || [];
          setRelatedSuggestions(mappedSuggestions);
        } catch (error) {
          console.error("Failed to get related suggestions:", error);
          setRelatedSuggestions([]);
        }
      }
    }
  };

  const generateContentBasedOnSelection = (
    selectedText: string,
    docName: string,
  ): string => {
    const text = selectedText.toLowerCase();

    if (text.includes("hydrogen")) {
      return `Analysis of hydrogen-related content: The selected text "${selectedText.substring(0, 50)}..." from ${docName} discusses hydrogen atomic properties. This relates to quantum mechanics, electronic structure, and spectroscopic analysis commonly found in physical chemistry assignments.`;
    } else if (text.includes("atom")) {
      return `Atomic structure analysis: The selected passage "${selectedText.substring(0, 50)}..." from ${docName} covers atomic theory concepts. This content is typically associated with quantum mechanical models and electron behavior in atoms.`;
    } else if (text.includes("energy")) {
      return `Energy analysis: The selected text "${selectedText.substring(0, 50)}..." from ${docName} discusses energy concepts. This may relate to quantum energy levels, electronic transitions, or thermodynamic principles.`;
    } else if (text.includes("wave") || text.includes("function")) {
      return `Wave function analysis: The selected content "${selectedText.substring(0, 50)}..." from ${docName} appears to discuss wave functions or mathematical expressions related to quantum mechanics.`;
    } else {
      return `Content analysis: The selected text "${selectedText.substring(0, 50)}..." from ${docName} contains scientific or mathematical content that may be relevant to the overall document themes and academic context.`;
    }
  };

  const getContextAroundSelection = (selection: Selection): string => {
    // Mock context extraction - in real implementation, this would extract surrounding text
    return "This is the context around the selected text for better AI understanding...";
  };

  const triggerAIInsights = async (text: string) => {
    if (!sessionId || !currentDocument) return;

    // Clear previous insights
    setAIInsights([]);
    setIsGeneratingInsights(true);

    try {
      // Call real backend AI insights API with timeout
      const { getAIInsights } = await import("../../../shared/api");

      // Race between API call and timeout
      const timeoutPromise = new Promise(
        (_, reject) => setTimeout(() => reject(new Error("Timeout")), 30000), // 30 second timeout
      );

      const insightResponse = await Promise.race([
        getAIInsights(
          sessionId,
          text,
          `Selected from: ${currentDocument.name}`,
        ),
        timeoutPromise,
      ]);

      console.log("Real backend response received:", insightResponse);
      console.log("Insights type:", typeof insightResponse.insights);
      console.log("Insights structure:", insightResponse.insights);

      // Convert backend response to frontend format
      const insight = {
        id: Math.random().toString(36).substr(2, 9),
        type: "summary" as const,
        content:
          typeof insightResponse.insights === "string"
            ? insightResponse.insights
            : insightResponse.insights?.overall_analysis ||
              `Analysis of "${text.substring(0, 50)}...": Analysis completed.`,
        confidence: 0.85,
        sourceDocuments: [currentDocument.id],
        relatedSections: (insightResponse.related_chunks || []).map(
          (chunk: any) => ({
            documentId: currentDocument.id,
            text: chunk.content?.substring(0, 100) + "..." || "Related content",
            page: chunk.page_number,
          }),
        ),
        // Add structured insights if available
        structured_insights:
          typeof insightResponse.insights === "object"
            ? insightResponse.insights
            : undefined,
      };

      console.log("Created insight object:", insight);
      console.log(
        "Structured insights available:",
        !!insight.structured_insights,
      );

      addAIInsight(insight);

      // Update journey bar to step 3 (insight generated)
      setCurrentStep(3);
      console.log(
        "[JourneyBar Debug] setCurrentStep(3) called after AI insight generated",
      );
      console.log("Backend insight response:", insightResponse);
    } catch (error) {
      console.error("Failed to generate AI insights:", error);

      // Fallback with content based on actual selected text
      const fallbackInsight = {
        id: Math.random().toString(36).substr(2, 9),
        type: "summary" as const,
        content: generateContentBasedOnSelection(text, currentDocument.name),
        confidence: 0.75,
        sourceDocuments: [currentDocument.id],
        relatedSections: [
          {
            documentId: currentDocument.id,
            text: text.substring(0, 150) + "...",
          },
        ],
      };

      addAIInsight(fallbackInsight);
      setCurrentStep(3);
    } finally {
      setIsGeneratingInsights(false);
    }
  };
  // Removed global selection handler to prevent conflicts with Adobe selection
  // Adobe PDF viewer handles text selection internally

  const animateConnectionThread = () => {
    // Create a visual connection line from selection to insights panel
    const thread = document.createElement("div");
    thread.className = "absolute bg-primary/50 h-px pointer-events-none";
    thread.style.width = "0px";
    thread.style.transformOrigin = "left center";

    const connectionContainer = document.getElementById("connection-thread");
    if (connectionContainer) {
      connectionContainer.appendChild(thread);

      gsap.to(thread, {
        width: "300px",
        duration: 0.8,
        ease: "power2.out",
        onComplete: () => {
          setTimeout(() => {
            gsap.to(thread, {
              opacity: 0,
              duration: 0.5,
              onComplete: () => connectionContainer.removeChild(thread),
            });
          }, 2000);
        },
      });
    }
  };

  if (!currentDocument) {
    return (
      <div className="flex-1 flex items-center justify-center bg-slate-900 text-white">
        <div className="text-center max-w-md">
          <FileText className="h-16 w-16 mx-auto mb-4 text-muted-foreground opacity-50" />
          <h3 className="text-lg font-medium mb-2">Select a Document</h3>
          <p className="text-muted-foreground text-sm">
            Choose a document from the left sidebar to start your research
            session.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className={cn("flex-1 flex flex-col bg-slate-900 min-h-0 h-full", className)}>
      {/* Document Viewer Header */}
      <div className="glass-panel border-b border-white/10 px-2 sm:px-4 py-2 sm:py-3">
        <div className="flex items-center gap-2 min-w-0 flex-1">
          <FileText className="h-4 w-4 sm:h-5 sm:w-5 text-primary flex-shrink-0" />
          <div className="min-w-0 flex-1">
            <h2 className="font-medium text-sm sm:text-base truncate">
              {currentDocument.name}
            </h2>
          </div>
        </div>
      </div>
      {/* PDF Embed Viewer */}
      <div className="flex-1 overflow-auto p-4 max-h-full min-h-0">
        {currentDocument?.url && (
          <AdobeEmbedViewer
            ref={adobeRef}
            fileName={currentDocument.name}
            docId={currentDocument.id}
            sessionId={sessionId || "default"}
            onSelection={handleAdobeSelection}
          />
        )}
      </div>
    </div>
  );
}
