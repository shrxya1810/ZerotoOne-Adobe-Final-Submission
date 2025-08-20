import { useState, useEffect } from "react";
import { useAppStore } from "../../lib/store";
import { analyzePersona } from "../../../shared/api";
import { Button } from "../ui/button";
import { Badge } from "../ui/badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "../ui/card";
import { Textarea } from "../ui/textarea";
import {
  Settings,
  User,
  MessageSquare,
  Microscope,
  GraduationCap,
  TrendingUp,
  Code,
  PenTool,
  Zap,
  ChevronUp,
  ChevronDown,
  Sparkles,
  CheckCircle2,
  FileText,
} from "lucide-react";
import { cn } from "../../lib/utils";

export default function PersonaPanel() {
  const {
    currentDocument,
    setDocumentConfiguration,
    getDocumentConfiguration,
    sessionId,
  } = useAppStore();

  const [isExpanded, setIsExpanded] = useState(true);
  const [selectedPersona, setSelectedPersona] = useState<string>("researcher");
  const [analysisPrompt, setAnalysisPrompt] = useState<string>("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResults, setAnalysisResults] = useState<any>(null);

  const personas = [
    {
      id: "researcher",
      name: "Researcher",
      icon: <Microscope className="h-5 w-5" />,
    },
    {
      id: "student",
      name: "Student",
      icon: <GraduationCap className="h-5 w-5" />,
    },
    {
      id: "analyst",
      name: "Analyst",
      icon: <TrendingUp className="h-5 w-5" />,
    },
    { id: "developer", name: "Developer", icon: <Code className="h-5 w-5" /> },
    { id: "writer", name: "Writer", icon: <PenTool className="h-5 w-5" /> },
    { id: "custom", name: "Custom", icon: <Settings className="h-5 w-5" /> },
  ];

  // Auto-fill prompt when persona changes
  useEffect(() => {
    if (selectedPersona && selectedPersona !== "custom") {
      setAnalysisPrompt(getSuggestedPrompt(selectedPersona));
    }
  }, [selectedPersona]);

  const getSuggestedPrompt = (personaId: string) => {
    const suggestions = {
      researcher:
        "Focus on methodology, findings, limitations, and implications. Identify gaps and propose next experiments.",
      student:
        "Summarize key concepts in simple terms, list learning objectives, and suggest study pointers.",
      analyst:
        "Extract trends, risks, ROI implications, and provide actionable recommendations with rationale.",
      developer:
        "Emphasize architecture, data flow, APIs, and constraints. List pitfalls, trade-offs, and code-relevant snippets.",
      writer:
        "Outline narrative arcs, key arguments, evidence, counterpoints, and fact-check targets.",
      custom:
        "Describe what you want. E.g., summarize security risks; compare approaches; list deployment steps.",
    };
    return (suggestions as any)[personaId] || suggestions.custom;
  };

  const handleApplyConfiguration = async () => {
    if (
      !selectedPersona ||
      !analysisPrompt.trim() ||
      !currentDocument ||
      !sessionId
    )
      return;

    setIsAnalyzing(true);
    try {
      const config = {
        documentId: currentDocument.id,
        persona: selectedPersona,
        prompt: analysisPrompt,
        appliedAt: new Date(),
      };
      setDocumentConfiguration(config);

      const personaName =
        personas.find((p) => p.id === selectedPersona)?.name || selectedPersona;

      // mpnet-backed processing lives behind this API on your backend
      const result = await analyzePersona(
        sessionId,
        personaName,
        analysisPrompt,
        5,
      );
      setAnalysisResults(result);
    } catch (error) {
      console.error("Persona analysis failed:", error);
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Navigation for relevant sections (similar to RightSidebar)
  const documents = useAppStore((s) => s.documents);
  const setCurrentDocument = useAppStore((s) => s.setCurrentDocument);
  const currentDocumentStore = useAppStore((s) => s.currentDocument);

  // Helper function to scroll to PDF viewer
  const scrollToPDFViewer = () => {
    try {
      // Look for the Adobe PDF viewer container
      const pdfViewer = document.querySelector('[id*="adobe-dc-view"]') || 
                        document.querySelector('.adobe-dc-view') ||
                        document.querySelector('[data-testid="pdf-viewer"]') ||
                        document.querySelector('iframe[src*="adobe"]');
      
      if (pdfViewer) {
        pdfViewer.scrollIntoView({ 
          behavior: 'smooth', 
          block: 'center' 
        });
        console.log('PersonaPanel: Scrolled to PDF viewer');
      } else {
        // Fallback: scroll to the document viewer component area
        const documentViewer = document.querySelector('[class*="DocumentViewer"]') ||
                              document.querySelector('.main-content') ||
                              document.querySelector('.center-panel');
        
        if (documentViewer) {
          documentViewer.scrollIntoView({ 
            behavior: 'smooth', 
            block: 'center' 
          });
          console.log('PersonaPanel: Scrolled to document viewer area');
        } else {
          // Final fallback: scroll to top of viewport
          window.scrollTo({ top: 0, behavior: 'smooth' });
          console.log('PersonaPanel: Scrolled to top as fallback');
        }
      }
    } catch (error) {
      console.warn('PersonaPanel: Error scrolling to PDF viewer:', error);
    }
  };

  const navigateToRelevantSection = (section: any) => {
    console.log(
      "PersonaPanel: navigateToRelevantSection called with:",
      section,
    );

    // Find the document by name (flexible matching)
    const targetDocument = documents?.find((doc) => {
      if (doc.name === section.document || doc.name === section.doc_title)
        return true;
      const docNameWithoutExt = doc.name.replace(/\.pdf$/i, "");
      const sectionNameWithoutExt = (
        section.document ||
        section.doc_title ||
        ""
      ).replace(/\.pdf$/i, "");
      if (docNameWithoutExt === sectionNameWithoutExt) return true;
      if (
        doc.name.includes(section.document || "") ||
        (section.document || "").includes(doc.name)
      )
        return true;
      if (
        docNameWithoutExt.includes(sectionNameWithoutExt) ||
        sectionNameWithoutExt.includes(docNameWithoutExt)
      )
        return true;
      return false;
    });

    if (!targetDocument) {
      console.warn(
        `PersonaPanel: Document "${section.document}" not found in available documents`,
      );
      console.log(
        "Available documents:",
        documents?.map((d) => d.name),
      );
      return;
    }

    console.log(`PersonaPanel: Found target document: ${targetDocument.name}`);

    // Extract page number - persona API already returns 1-based page numbers
    const pageNumber =
      typeof section.page_number === "number"
        ? section.page_number
        : section.page;

    // Persona API returns 1-based page numbers, use directly for Adobe viewer
    const validPageNumber = Math.max(1, typeof pageNumber === "number" ? pageNumber : 1);

    console.log(
      `PersonaPanel: Using page number ${pageNumber} (already 1-based from persona API) for Adobe viewer`,
    );

    // Check if we need to switch documents
    const isDocumentSwitch =
      !currentDocumentStore || currentDocumentStore.id !== targetDocument.id;

    if (isDocumentSwitch) {
      setCurrentDocument(targetDocument);
      console.log(`PersonaPanel: Switched to document: ${targetDocument.name}`);
    }

    // Wait for document to load if switching, then navigate with retry logic
    const delay = isDocumentSwitch ? 1500 : 200;

    const dispatchNavigation = (attemptCount = 0) => {
      const maxAttempts = 5;

      try {
        window.dispatchEvent(
          new CustomEvent("go-to-page", {
            detail: { page: validPageNumber },
            bubbles: true,
          }),
        );
        
        // Scroll to PDF viewer after navigation
        scrollToPDFViewer();
        
        console.log(
          `PersonaPanel: Dispatched go-to-page event with page ${validPageNumber} (1-based) - attempt ${attemptCount + 1}`,
        );
      } catch (error) {
        console.error(
          "PersonaPanel: Error dispatching navigation event:",
          error,
        );

        // Retry with exponential backoff
        if (attemptCount < maxAttempts) {
          setTimeout(
            () => dispatchNavigation(attemptCount + 1),
            Math.pow(2, attemptCount) * 200,
          );
        }
      }
    };

    setTimeout(() => {
      dispatchNavigation();
    }, delay);

    console.log(
      `PersonaPanel: Scheduled navigation to page ${validPageNumber} (1-based) with ${delay}ms delay${isDocumentSwitch ? " (document switch)" : ""}`,
    );
  };

  if (!currentDocument) return null;

  const activeConfig = getDocumentConfiguration(currentDocument.id);

  return (
    <div className="bg-black text-white border border-red-800/40 rounded-lg">
      {/* Header */}
      <div className="flex items-center justify-between p-3 border-b border-red-800/30">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-lg bg-red-700/20 border border-red-700/40">
            <Sparkles className="h-4 w-4 text-red-400" />
          </div>
          <div>
            <h3 className="text-sm font-semibold">Persona Analysis</h3>
            <p className="text-xs text-red-200/70">
              AI analysis tailored to your role
            </p>
          </div>
        </div>
        <Button
          variant="ghost"
          size="sm"
          onClick={() => setIsExpanded(!isExpanded)}
          className="h-8 w-8 p-0 hover:bg-red-900/30 rounded-lg"
        >
          {isExpanded ? (
            <ChevronDown className="h-3 w-3" />
          ) : (
            <ChevronUp className="h-3 w-3" />
          )}
        </Button>
      </div>

      {/* Body */}
      {isExpanded && (
        <div className="p-6 space-y-6">
          {/* Input boxes */}
          <div className="space-y-6">
            {/* Persona Selection */}
            <Card className="bg-black border border-red-800/40">
              <CardHeader className="pb-3">
                <CardTitle className="text-base flex items-center gap-2">
                  <User className="h-5 w-5 text-red-400" />
                  Choose Persona
                </CardTitle>
                <CardDescription className="text-sm text-red-200/80">
                  Select your role for tailored analysis
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                  {personas.map((persona) => {
                    const active = selectedPersona === persona.id;
                    return (
                      <Button
                        key={persona.id}
                        variant={active ? "default" : "outline"}
                        onClick={() => setSelectedPersona(persona.id)}
                        className={cn(
                          "h-auto min-h-[72px] p-3 text-sm flex flex-col items-center justify-center gap-2 rounded-xl border",
                          active
                            ? "bg-gradient-to-br from-red-700 to-red-600 text-white border-red-500 shadow-lg"
                            : "bg-black text-white border-red-800/40 hover:border-red-600/60 hover:bg-red-900/20",
                        )}
                      >
                        <div
                          className={cn(
                            "p-2 rounded-md",
                            active ? "bg-white/15" : "bg-red-900/30",
                          )}
                        >
                          {persona.icon}
                        </div>
                        <span className="font-semibold">{persona.name}</span>
                      </Button>
                    );
                  })}
                </div>
              </CardContent>
            </Card>

            {/* Prompt BELOW persona */}
            <Card className="bg-black border border-red-800/40">
              <CardHeader className="pb-3">
                <CardTitle className="text-base flex items-center gap-2">
                  <MessageSquare className="h-5 w-5 text-red-400" />
                  Analysis Prompt
                </CardTitle>
                <CardDescription className="text-sm text-red-200/80">
                  Describe what you want from the documents
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  <Textarea
                    placeholder="e.g., Compare methods, extract risks, list deployment steps…"
                    value={analysisPrompt}
                    onChange={(e) => setAnalysisPrompt(e.target.value)}
                    className="min-h-[120px] text-sm bg-black border-red-800/50 focus:border-red-500 focus:ring-red-500/30 text-white"
                  />
                  <Button
                    onClick={handleApplyConfiguration}
                    className="w-full h-11 bg-gradient-to-r from-red-700 via-red-600 to-red-700 hover:from-red-600 hover:to-red-500 text-white font-semibold rounded-lg"
                    disabled={
                      !selectedPersona || !analysisPrompt.trim() || isAnalyzing
                    }
                  >
                    <Zap className="h-5 w-5 mr-2" />
                    {isAnalyzing ? "Analyzing…" : "Run Persona Analysis"}
                  </Button>
                  <p className="text-xs text-red-200/70 text-center">
                    Uses MPNet-backed processing in the backend for retrieval &
                    insights.
                  </p>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Output panel below input boxes */}
          <div className="mt-6">
            <Card className="bg-black border border-red-800/40">
              <CardHeader className="pb-3">
                <CardTitle className="text-base flex items-center gap-2">
                  <FileText className="h-5 w-5 text-red-400" />
                  Persona Output
                </CardTitle>
                <CardDescription className="text-sm text-red-200/80">
                  Summary and relevant sections
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {!analysisResults && (
                  <p className="text-sm text-red-200/70">
                    Run an analysis to see the persona‑focused summary and top
                    sections here.
                  </p>
                )}

                {/* Summary */}
                {analysisResults?.summary && (
                  <div className="p-3 rounded-lg bg-black border border-red-800/40">
                    <h4 className="text-sm font-semibold mb-1">Summary</h4>
                    <p className="text-sm text-red-100/90 leading-relaxed">
                      {analysisResults.summary}
                    </p>
                  </div>
                )}

                {/* Recommendations (optional) */}
                {analysisResults?.recommendations?.length > 0 && (
                  <div className="p-3 rounded-lg bg-black border border-red-800/40">
                    <h4 className="text-sm font-semibold mb-1">
                      Recommendations
                    </h4>
                    <ul className="space-y-1">
                      {analysisResults.recommendations.map(
                        (rec: string, i: number) => (
                          <li
                            key={i}
                            className="text-sm text-red-100/90 flex items-start gap-2"
                          >
                            <span className="text-red-400 mt-1">•</span>
                            {rec}
                          </li>
                        ),
                      )}
                    </ul>
                  </div>
                )}

                {/* Information Gaps */}
                {analysisResults?.gaps?.length > 0 && (
                  <div className="p-3 rounded-lg bg-black border border-orange-800/40">
                    <h4 className="text-sm font-semibold mb-1 text-orange-200">
                      Information Gaps
                    </h4>
                    <ul className="space-y-1">
                      {analysisResults.gaps.map((gap: string, i: number) => (
                        <li
                          key={i}
                          className="text-sm text-orange-100/90 flex items-start gap-2"
                        >
                          <span className="text-orange-400 mt-1">⚠</span>
                          {gap}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}

                {/* Relevant sections with Go To */}
                {analysisResults?.extracted_sections?.length > 0 && (
                  <div className="space-y-2">
                    <h4 className="text-sm font-semibold">Relevant Sections</h4>
                    {analysisResults.extracted_sections.map(
                      (s: any, idx: number) => (
                        <div
                          key={`${s.doc_id || s.document}-${idx}`}
                          className="p-3 rounded-lg bg-black border border-red-800/40"
                        >
                          <div className="flex items-center justify-between gap-3">
                            <div className="min-w-0">
                              <p className="text-xs text-red-200/80 truncate">
                                {s.document || s.doc_title || "Document"} • Page{" "}
                                {typeof s.page_number === "number"
                                  ? s.page_number
                                  : s.page}
                              </p>
                              <h5 className="text-sm font-medium">
                                {s.section_title || s.title}
                              </h5>
                            </div>
                            <Badge
                              variant="outline"
                              className="text-[10px] border-red-700/60 text-red-300"
                            >
                              {(s.relevance_score != null
                                ? Math.round(s.relevance_score * 100)
                                : s.score != null
                                  ? Math.round(s.score * 100)
                                  : 0) + "%"}
                            </Badge>
                          </div>
                          {s.content && (
                            <div className="text-xs text-red-100/80 mt-2 leading-relaxed">
                              {s.content}
                            </div>
                          )}
                          <div className="mt-3 flex items-center gap-2">
                            <Button
                              variant="outline"
                              className="h-8 px-3 border-red-700/60 text-white hover:bg-red-900/30"
                              onClick={() => navigateToRelevantSection(s)}
                            >
                              Go to in PDF
                            </Button>
                            {typeof s.importance_rank === "number" && (
                              <Badge className="bg-red-700/30 text-red-200 border border-red-700/50">
                                Rank #{s.importance_rank}
                              </Badge>
                            )}
                          </div>
                        </div>
                      ),
                    )}
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </div>
      )}
    </div>
  );
}
