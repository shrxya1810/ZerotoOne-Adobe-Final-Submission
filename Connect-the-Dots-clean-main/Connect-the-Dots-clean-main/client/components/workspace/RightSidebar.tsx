import { useState, useRef, useEffect } from "react";
import { useAppStore } from "../../lib/store";
import { Button } from "../ui/button";
import { ScrollArea } from "../ui/scroll-area";
import { Badge } from "../ui/badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "../ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "../ui/tabs";
import { Textarea } from "../ui/textarea";
import {
  Brain,
  Search,
  Volume2,
  MessageCircle,
  Lightbulb,
  ExternalLink,
  Copy,
  ThumbsUp,
  ThumbsDown,
  Play,
  Pause,
  Download,
  Zap,
  Eye,
  ChevronRight,
} from "lucide-react";
import { cn } from "../../lib/utils";
import gsap from "gsap";
import {
  generatePodcastAudio,
  analyzePersona,
  getPersonaSuggestions,
  askAI,
} from "../../../shared/api";
import { RelatedSuggestion } from "../../lib/store";

interface AIInsightCard {
  id: string;
  type: "summary" | "related" | "contradiction" | "enhancement";
  title: string;
  content: string;
  confidence: number;
}

export default function RightSidebar() {
  const {
    currentSelection,
    aiInsights,
    isGeneratingInsights,
    isGeneratingAudio,
    setIsGeneratingAudio,
    setCurrentAudioUrl,
    currentAudioUrl,
    relatedSuggestions,
    sessionId,
    documents,
    currentDocument,
    setCurrentDocument,
  } = useAppStore();

  const [activeTab, setActiveTab] = useState("insights");
  const [chatInput, setChatInput] = useState("");
  const [isAudioPlaying, setIsAudioPlaying] = useState(false);
  const [chatHistory, setChatHistory] = useState<
    Array<{ id: string; question: string; answer: string; sources?: any[] }>
  >([]);
  const [isAsking, setIsAsking] = useState(false);
  const audioRef = useRef<HTMLAudioElement>(null);
  const [audioDuration, setAudioDuration] = useState<string>("0:00");
  const [audioProgress, setAudioProgress] = useState<number>(0);

  // Convert backend AI insights to display format
  const displayInsights = aiInsights.map((insight) => ({
    id: insight.id,
    type: insight.type as AIInsightCard["type"],
    title: insight.type === "summary" ? "Summary" : "Analysis",
    content: insight.content,
    confidence: insight.confidence,
  }));

  const insightCardRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (currentSelection && insightCardRef.current) {
      // Animate insight cards when new selection is made
      gsap.fromTo(
        insightCardRef.current.children,
        { opacity: 0, x: 20 },
        { opacity: 1, x: 0, duration: 0.5, stagger: 0.1, ease: "power2.out" },
      );
    }
  }, [currentSelection]);

  useEffect(() => {
    // Reset audio state and show generate button when selection or document changes
    setCurrentAudioUrl(null);
    setIsAudioPlaying(false);
    setAudioDuration("0:00");
    setAudioProgress(0);
  }, [currentSelection, currentDocument]);

  const generateAudioOverview = async () => {
    if (!currentSelection || !sessionId) return;

    setIsGeneratingAudio(true);

    try {
      // Generate audio overview from selected text
      console.log("Generating audio for session:", sessionId);
      const result = await generatePodcastAudio(sessionId, {
        topic: `Analysis of: ${currentSelection.text.substring(0, 100)}...`,
        voice: "en-US-AriaNeural",
        tts_provider: "azure",
      });

      console.log("Audio generation result:", result);

      if (result.audio_file_path) {
        // For simplicity, always use a generic URL that will return the most recent file
        const audioUrl = `/api/podcast/audio/${sessionId}/latest.wav`;
        console.log("Setting audio URL:", audioUrl);
        setCurrentAudioUrl(audioUrl);
      } else {
        console.error("No audio file path in response:", result);
      }
    } catch (error) {
      console.error("Audio generation failed:", error);
      // Fallback to mock for demo
      const mockAudioUrl = "data:audio/wav;base64,mock-audio-data";
      setCurrentAudioUrl(mockAudioUrl);
    } finally {
      setIsGeneratingAudio(false);
    }
  };

  const toggleAudioPlayback = () => {
    if (!audioRef.current || !currentAudioUrl) return;

    if (isAudioPlaying) {
      audioRef.current.pause();
      setIsAudioPlaying(false);
    } else {
      audioRef.current.play();
      setIsAudioPlaying(true);
    }
  };

  const handleAudioLoadedMetadata = () => {
    if (audioRef.current) {
      const duration = audioRef.current.duration;
      const minutes = Math.floor(duration / 60);
      const seconds = Math.floor(duration % 60);
      setAudioDuration(`${minutes}:${seconds.toString().padStart(2, "0")}`);
    }
  };

  const handleAudioTimeUpdate = () => {
    if (audioRef.current) {
      const progress =
        (audioRef.current.currentTime / audioRef.current.duration) * 100;
      setAudioProgress(isNaN(progress) ? 0 : progress);
    }
  };

  const handleAudioEnded = () => {
    setIsAudioPlaying(false);
    setAudioProgress(0);
  };

  const handleAskQuestion = async () => {
    if (!chatInput.trim() || !sessionId || isAsking) return;

    const question = chatInput.trim();
    setChatInput("");
    setIsAsking(true);

    try {
      const response = await askAI(sessionId, question, true);

      const chatItem = {
        id: `chat_${Date.now()}`,
        question,
        answer: response.answer || "No answer provided",
        sources: response.sources || [],
      };

      setChatHistory((prev) => [...prev, chatItem]);
    } catch (error) {
      console.error("Ask AI failed:", error);
      const errorItem = {
        id: `chat_${Date.now()}`,
        question,
        answer:
          "Sorry, I encountered an error while processing your question. Please try again.",
        sources: [],
      };
      setChatHistory((prev) => [...prev, errorItem]);
    } finally {
      setIsAsking(false);
    }
  };

  const navigateToSection = (section: RelatedSuggestion) => {
    // Extract document name from different possible fields
    const documentName =
      section.documentName ||
      (section as any).document ||
      (section as any).doc_title ||
      (section as any).doc_name;

    // Extract page number from different possible fields
    let pageNumber =
      section.page !== undefined
        ? section.page
        : (section as any).page_number !== undefined
          ? (section as any).page_number
          : (section as any).pageNumber !== undefined
            ? (section as any).pageNumber
            : undefined;

    // Keep original page number as-is - the backend should already provide correct 0-based indexing
    // No conversion needed since pageNumber should be 0-based from the backend

    console.log(
      `Navigating to "${documentName}", page ${pageNumber !== undefined ? pageNumber + 1 : "unknown"} (0-indexed: ${pageNumber})`,
    );

    if (!documentName) {
      console.warn("No document name found in section:", section);
      return;
    }

    // Find the document by name with more robust matching
    const targetDocument = documents.find((doc) => {
      // First try exact match
      if (doc.name === documentName) return true;

      // Try removing .pdf extension for comparison
      const docNameWithoutExt = doc.name.replace(/\.pdf$/i, "");
      const sectionNameWithoutExt = documentName.replace(/\.pdf$/i, "");
      if (docNameWithoutExt === sectionNameWithoutExt) return true;

      // Try case-insensitive exact match
      if (doc.name.toLowerCase() === documentName.toLowerCase()) return true;
      if (
        docNameWithoutExt.toLowerCase() === sectionNameWithoutExt.toLowerCase()
      )
        return true;

      // Try partial matches (both directions)
      if (
        doc.name.toLowerCase().includes(documentName.toLowerCase()) ||
        documentName.toLowerCase().includes(doc.name.toLowerCase())
      )
        return true;

      if (
        docNameWithoutExt
          .toLowerCase()
          .includes(sectionNameWithoutExt.toLowerCase()) ||
        sectionNameWithoutExt
          .toLowerCase()
          .includes(docNameWithoutExt.toLowerCase())
      )
        return true;

      // Try matching by removing common prefixes/suffixes
      const cleanDocName = docNameWithoutExt
        .replace(/^(the\s+|a\s+|\d+\.\s*)/i, "")
        .trim();
      const cleanSectionName = sectionNameWithoutExt
        .replace(/^(the\s+|a\s+|\d+\.\s*)/i, "")
        .trim();
      if (cleanDocName.toLowerCase() === cleanSectionName.toLowerCase())
        return true;

      return false;
    });

    if (!targetDocument) {
      console.warn(
        `Document "${documentName}" not found in available documents`,
      );
      console.log(
        "Available documents:",
        documents.map((d) => d.name),
      );
      return;
    }

    console.log(`Found target document: ${targetDocument.name}`);

    // If it's a different document, switch to it first
    const switchingDocs = !currentDocument || currentDocument.id !== targetDocument.id;
    if (switchingDocs) {
      setCurrentDocument(targetDocument);
      console.log(`Switched to document: ${targetDocument.name}`);
    }

    if (pageNumber !== undefined) {
      // Backend provides 0-based page numbers, Adobe viewer expects 1-based
      // Simply add 1 to convert from 0-based to 1-based
      const validPageNumber = Math.max(1, pageNumber + 1);
      console.log(
        `Converting page ${pageNumber} (0-based) to ${validPageNumber} (1-based) for Adobe viewer`,
      );

      // Robustly retry go-to-page event after switching documents
      const maxAttempts = 8;
      let attempts = 0;
      const tryDispatch = () => {
        attempts++;
        try {
          window.dispatchEvent(
            new CustomEvent("go-to-page", {
              detail: { page: validPageNumber },
              bubbles: true,
            })
          );
          console.log(`Dispatched go-to-page event with page ${validPageNumber} (1-based for Adobe viewer), attempt ${attempts}`);
        } catch (error) {
          console.error("Error dispatching navigation event:", error);
        }
        // If switching docs, retry a few times to ensure viewer is ready
        if (switchingDocs && attempts < maxAttempts) {
          setTimeout(tryDispatch, 350);
        }
      };
      // If switching docs, start after a longer delay, else short delay
      setTimeout(tryDispatch, switchingDocs ? 900 : 150);
      console.log(
        `Scheduled navigation to page ${validPageNumber} (1-based) with retry logic (switchingDocs=${switchingDocs})`,
      );
    } else {
      console.log("No page number provided, switching document only");
    }
  };

  const copyInsight = (content: string) => {
    navigator.clipboard.writeText(content);
  };

  const getInsightIcon = (type: string) => {
    switch (type) {
      case "summary":
        return <Brain className="h-4 w-4" />;
      case "related":
        return <Search className="h-4 w-4" />;
      case "contradiction":
        return <Lightbulb className="h-4 w-4" />;
      case "enhancement":
        return <Zap className="h-4 w-4" />;
      default:
        return <Brain className="h-4 w-4" />;
    }
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.9) return "text-green-400";
    if (confidence >= 0.8) return "text-yellow-400";
    return "text-orange-400";
  };

  return (
    <div className="h-full flex flex-col bg-panel">
      {/* Right Sidebar Header */}
      <div className="p-4 border-b border-border">
        <h3 className="font-medium text-sm mb-3">AI Research Assistant</h3>

        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="grid w-full grid-cols-3 bg-workspace">
            <TabsTrigger value="insights" className="text-xs">
              <Brain className="h-3 w-3 mr-1" />
              Insights
            </TabsTrigger>
            <TabsTrigger value="related" className="text-xs">
              <Search className="h-3 w-3 mr-1" />
              Related
            </TabsTrigger>
            <TabsTrigger value="chat" className="text-xs">
              <MessageCircle className="h-3 w-3 mr-1" />
              Chat
            </TabsTrigger>
          </TabsList>
        </Tabs>
      </div>

      {/* Sidebar Content */}
      <ScrollArea className="flex-1">
        <div className="p-4">
          <Tabs value={activeTab} className="w-full">
            {/* AI Insights Tab */}
            <TabsContent value="insights" className="space-y-4 mt-0">
              {currentSelection ? (
                <>
                  {/* Audio Overview */}
                  <Card className="bg-workspace border-border">
                    <CardHeader className="pb-3">
                      <CardTitle className="text-sm flex items-center gap-2">
                        <Volume2 className="h-4 w-4" />
                        Audio Overview
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-3">
                      {isGeneratingAudio ? (
                        <div className="flex items-center gap-2 text-sm text-muted-foreground">
                          <div className="animate-spin rounded-full h-4 w-4 border-2 border-primary border-t-transparent"></div>
                          Generating conversational overview...
                        </div>
                      ) : currentAudioUrl ? (
                        <div className="space-y-2">
                          <div className="flex items-center gap-2">
                            <Button
                              variant="outline"
                              size="sm"
                              onClick={toggleAudioPlayback}
                              className="flex-1"
                            >
                              {isAudioPlaying ? (
                                <Pause className="h-3 w-3 mr-1" />
                              ) : (
                                <Play className="h-3 w-3 mr-1" />
                              )}
                              {isAudioPlaying ? "Pause" : "Play"}
                            </Button>
                            <Button variant="ghost" size="sm">
                              <Download className="h-3 w-3" />
                            </Button>
                          </div>
                          <div className="w-full bg-muted rounded-full h-2">
                            <div
                              className="bg-primary h-2 rounded-full transition-all duration-300"
                              style={{ width: `${audioProgress}%` }}
                            ></div>
                          </div>
                          <p className="text-xs text-muted-foreground">
                            AI-generated overview • {audioDuration} duration
                          </p>
                          <audio
                            ref={audioRef}
                            src={currentAudioUrl}
                            onLoadedMetadata={handleAudioLoadedMetadata}
                            onTimeUpdate={handleAudioTimeUpdate}
                            onEnded={handleAudioEnded}
                            style={{ display: "none" }}
                          />
                        </div>
                      ) : (
                        <Button
                          onClick={generateAudioOverview}
                          variant="outline"
                          size="sm"
                          className="w-full"
                        >
                          <Volume2 className="h-3 w-3 mr-1" />
                          Generate Audio Overview
                        </Button>
                      )}
                    </CardContent>
                  </Card>

                  {/* AI Insights - Three Structured Boxes */}
                  <div ref={insightCardRef} className="space-y-4">
                    {isGeneratingInsights ? (
                      <Card className="bg-workspace border-border">
                        <CardContent className="p-4">
                          <div className="flex items-center gap-2 text-sm text-muted-foreground">
                            <div className="animate-spin rounded-full h-4 w-4 border-2 border-primary border-t-transparent"></div>
                            Analyzing selection across all documents...
                          </div>
                        </CardContent>
                      </Card>
                    ) : aiInsights.length > 0 ? (
                      <>
                        {/* Check for simplified insights first, then fallback to structured */}
                        {aiInsights[0]?.simplified_insights ? (
                          <Card className="bg-workspace border-border">
                            <CardHeader className="pb-3">
                              <CardTitle className="text-sm flex items-center gap-2">
                                <Brain className="h-4 w-4" />
                                📄 Insights Analysis
                              </CardTitle>
                            </CardHeader>
                            <CardContent className="space-y-4">
                              <div>
                                <h4 className="text-xs font-semibold mb-2">
                                  Summary of Selection
                                </h4>
                                <p className="text-xs text-muted-foreground">
                                  {aiInsights[0].simplified_insights.summary}
                                </p>
                              </div>

                              <div>
                                <h4 className="text-xs font-semibold mb-2">
                                  Sentiment / Tone
                                </h4>
                                <Badge variant="outline" className="text-xs">
                                  {
                                    aiInsights[0].simplified_insights
                                      .sentiment_tone
                                  }
                                </Badge>
                              </div>

                              {aiInsights[0].simplified_insights
                                .contradictions !== "None detected" && (
                                <div>
                                  <h4 className="text-xs font-semibold mb-2">
                                    Contradictions
                                  </h4>
                                  <p className="text-xs text-muted-foreground">
                                    {
                                      aiInsights[0].simplified_insights
                                        .contradictions
                                    }
                                  </p>
                                </div>
                              )}

                              {aiInsights[0].simplified_insights
                                .overlap_redundancy !== "None found" && (
                                <div>
                                  <h4 className="text-xs font-semibold mb-2">
                                    Overlap / Redundancy
                                  </h4>
                                  <p className="text-xs text-muted-foreground">
                                    {
                                      aiInsights[0].simplified_insights
                                        .overlap_redundancy
                                    }
                                  </p>
                                </div>
                              )}

                              <div>
                                <h4 className="text-xs font-semibold mb-2">
                                  Supporting Evidence
                                </h4>
                                <p className="text-xs text-muted-foreground">
                                  {
                                    aiInsights[0].simplified_insights
                                      .supporting_evidence
                                  }
                                </p>
                              </div>

                              <div>
                                <h4 className="text-xs font-semibold mb-2">
                                  External Context & Sources
                                </h4>
                                <p className="text-xs text-muted-foreground">
                                  {
                                    aiInsights[0].simplified_insights
                                      .external_context
                                  }
                                </p>
                              </div>
                            </CardContent>
                          </Card>
                        ) : aiInsights[0]?.structured_insights ? (
                          <>
                            {/* Single Simplified Box */}
                            <Card className="bg-workspace border-border">
                              <CardHeader className="pb-3">
                                <CardTitle className="text-sm flex items-center gap-2">
                                  <Brain className="h-4 w-4" />
                                  📄 Analysis Summary
                                </CardTitle>
                              </CardHeader>
                              <CardContent className="space-y-4">
                                <div>
                                  <h4 className="text-xs font-semibold mb-2">
                                    Summary of Selection
                                  </h4>
                                  <p className="text-xs text-muted-foreground">
                                    {
                                      aiInsights[0].structured_insights
                                        .document_context.summary
                                    }
                                  </p>
                                </div>

                                <div>
                                  <h4 className="text-xs font-semibold mb-2">
                                    Sentiment / Tone
                                  </h4>
                                  <Badge variant="outline" className="text-xs">
                                    {
                                      aiInsights[0].structured_insights
                                        .document_context.sentiment_tone
                                    }
                                  </Badge>
                                </div>

                                {aiInsights[0].structured_insights
                                  .document_context.contradictions !==
                                  "None detected" &&
                                  aiInsights[0].structured_insights
                                    .document_context.contradictions !==
                                    "None identified" && (
                                    <div>
                                      <h4 className="text-xs font-semibold mb-2">
                                        Contradictions
                                      </h4>
                                      <p className="text-xs text-muted-foreground">
                                        {
                                          aiInsights[0].structured_insights
                                            .document_context.contradictions
                                        }
                                      </p>
                                    </div>
                                  )}

                                {aiInsights[0].structured_insights
                                  .document_context.overlap_redundancy !==
                                  "None found" &&
                                  aiInsights[0].structured_insights
                                    .document_context.overlap_redundancy !==
                                    "None" && (
                                    <div>
                                      <h4 className="text-xs font-semibold mb-2">
                                        Overlap / Redundancy
                                      </h4>
                                      <p className="text-xs text-muted-foreground">
                                        {
                                          aiInsights[0].structured_insights
                                            .document_context.overlap_redundancy
                                        }
                                      </p>
                                    </div>
                                  )}

                                <div>
                                  <h4 className="text-xs font-semibold mb-2">
                                    Supporting Evidence
                                  </h4>
                                  <p className="text-xs text-muted-foreground">
                                    {
                                      aiInsights[0].structured_insights
                                        .document_context.supporting_evidence
                                    }
                                  </p>
                                </div>

                                <div>
                                  <h4 className="text-xs font-semibold mb-2">
                                    External Context & Sources
                                  </h4>
                                  <div className="space-y-2">
                                    {aiInsights[0].structured_insights
                                      .external_context
                                      .external_contradictions && (
                                      <p className="text-xs text-muted-foreground">
                                        <span className="font-medium">
                                          External Contradictions:
                                        </span>{" "}
                                        {
                                          aiInsights[0].structured_insights
                                            .external_context
                                            .external_contradictions
                                        }
                                      </p>
                                    )}
                                    {aiInsights[0].structured_insights
                                      .external_context.counterpoints && (
                                      <p className="text-xs text-muted-foreground">
                                        <span className="font-medium">
                                          Counterpoints:
                                        </span>{" "}
                                        {
                                          aiInsights[0].structured_insights
                                            .external_context.counterpoints
                                        }
                                      </p>
                                    )}
                                    {aiInsights[0].structured_insights
                                      .external_context.real_world_examples && (
                                      <p className="text-xs text-muted-foreground">
                                        <span className="font-medium">
                                          Real-World Examples:
                                        </span>{" "}
                                        {
                                          aiInsights[0].structured_insights
                                            .external_context
                                            .real_world_examples
                                        }
                                      </p>
                                    )}
                                    {aiInsights[0].structured_insights
                                      .external_context.surprising_insights && (
                                      <p className="text-xs text-muted-foreground">
                                        <span className="font-medium">
                                          Surprising Insights:
                                        </span>{" "}
                                        {
                                          aiInsights[0].structured_insights
                                            .external_context
                                            .surprising_insights
                                        }
                                      </p>
                                    )}
                                    {aiInsights[0].structured_insights
                                      .external_context.related_articles && (
                                      <p className="text-xs text-muted-foreground">
                                        <span className="font-medium">
                                          Related Articles:
                                        </span>{" "}
                                        {
                                          aiInsights[0].structured_insights
                                            .external_context.related_articles
                                        }
                                      </p>
                                    )}
                                  </div>
                                </div>
                              </CardContent>
                            </Card>
                          </>
                        ) : (
                          // Fallback for old format insights
                          displayInsights.map((insight, index) => (
                            <Card
                              key={`${insight.id || "insight"}-${insight.type || "unknown"}-${index}`}
                              className="bg-workspace border-border hover:border-primary/50 transition-colors"
                            >
                              <CardHeader className="pb-3">
                                <div className="flex items-center justify-between">
                                  <CardTitle className="text-sm flex items-center gap-2">
                                    {getInsightIcon(insight.type)}
                                    {insight.title}
                                  </CardTitle>
                                  <div className="flex items-center gap-1">
                                    <Badge
                                      variant="outline"
                                      className={cn(
                                        "text-xs",
                                        getConfidenceColor(insight.confidence),
                                      )}
                                    >
                                      {(insight.confidence * 100).toFixed(0)}%
                                    </Badge>
                                    <Button
                                      variant="ghost"
                                      size="sm"
                                      onClick={() =>
                                        copyInsight(insight.content)
                                      }
                                    >
                                      <Copy className="h-3 w-3" />
                                    </Button>
                                  </div>
                                </div>
                              </CardHeader>
                              <CardContent>
                                <p className="text-sm text-muted-foreground mb-3">
                                  {insight.content}
                                </p>
                                <div className="flex items-center gap-2">
                                  <Button variant="ghost" size="sm">
                                    <ThumbsUp className="h-3 w-3" />
                                  </Button>
                                  <Button variant="ghost" size="sm">
                                    <ThumbsDown className="h-3 w-3" />
                                  </Button>
                                  <Button variant="ghost" size="sm">
                                    <ExternalLink className="h-3 w-3" />
                                  </Button>
                                </div>
                              </CardContent>
                            </Card>
                          ))
                        )}
                      </>
                    ) : (
                      <Card className="bg-workspace border-border">
                        <CardContent className="p-4">
                          <div className="text-center text-muted-foreground">
                            <Brain className="h-8 w-8 mx-auto mb-2 opacity-50" />
                            <p className="text-sm">
                              No AI insights generated yet
                            </p>
                          </div>
                        </CardContent>
                      </Card>
                    )}
                  </div>
                </>
              ) : (
                <div className="text-center text-muted-foreground">
                  <Eye className="h-12 w-12 mx-auto mb-3 opacity-50" />
                  <p className="text-sm">
                    Select text in the document to get AI insights
                  </p>
                </div>
              )}
            </TabsContent>

            {/* Related Sections Tab */}
            <TabsContent value="related" className="space-y-3 mt-0">
              {currentSelection ? (
                <>
                  <Card className="bg-workspace border-border">
                    <CardHeader className="pb-3">
                      <CardTitle className="text-sm">
                        Relevant Sections
                      </CardTitle>
                      <CardDescription className="text-xs">
                        Found {relatedSuggestions.filter(section => {
                          // Filter out suggestions with invalid page numbers
                          if (section.page === undefined) return true; // Keep if no page specified
                          
                          // Find the document this suggestion belongs to
                          const targetDoc = documents.find(doc => 
                            doc.name === section.documentName ||
                            doc.name.replace(/\.pdf$/i, '') === section.documentName.replace(/\.pdf$/i, '')
                          );
                          
                          // If we can't find the document, keep the suggestion but log warning
                          if (!targetDoc) {
                            console.warn(`Cannot validate page ${section.page} - document ${section.documentName} not found`);
                            return true;
                          }
                          
                          // For now, allow all pages since we don't have reliable page count
                          // TODO: Add page count validation when available
                          return true;
                        }).length} related passages
                      </CardDescription>
                    </CardHeader>
                  </Card>

                  {relatedSuggestions
                    .filter(section => {
                      // Filter out suggestions with invalid page numbers
                      if (section.page === undefined) return true; // Keep if no page specified
                      
                      // Find the document this suggestion belongs to
                      const targetDoc = documents.find(doc => 
                        doc.name === section.documentName ||
                        doc.name.replace(/\.pdf$/i, '') === section.documentName.replace(/\.pdf$/i, '')
                      );
                      
                      // If we can't find the document, keep the suggestion but log warning
                      if (!targetDoc) {
                        console.warn(`Cannot validate page ${section.page} - document ${section.documentName} not found`);
                        return true;
                      }
                      
                      // For now, allow all pages since we don't have reliable page count
                      // TODO: Add page count validation when available
                      return true;
                    })
                    .map((section, index) => (
                    <Card
                      key={`${section.id || section.documentName}-${section.page || 0}-${index}`}
                      className="bg-workspace border-border hover:border-primary/50 transition-colors cursor-pointer"
                      onClick={() => navigateToSection(section)}
                    >
                      <CardHeader className="pb-3">
                        <div className="flex items-center justify-between">
                          <CardTitle className="text-sm truncate">
                            {section.documentName}
                          </CardTitle>
                          <div className="flex items-center gap-1">
                            <Badge
                              variant="outline"
                              className={cn(
                                "text-xs",
                                getConfidenceColor(section.confidence),
                              )}
                            >
                              {(section.confidence * 100).toFixed(0)}%
                            </Badge>
                            <ChevronRight className="h-3 w-3" />
                          </div>
                        </div>
                      </CardHeader>
                      <CardContent>
                        <p className="text-sm text-muted-foreground mb-2">
                          {section.text}
                        </p>
                        {section.page !== undefined && (
                          <p className="text-xs text-muted-foreground">
                            Page {section.page + 1}
                          </p>
                        )}
                      </CardContent>
                    </Card>
                  ))}
                </>
              ) : (
                <div className="text-center text-muted-foreground">
                  <Search className="h-12 w-12 mx-auto mb-3 opacity-50" />
                  <p className="text-sm">
                    Select text to find related sections
                  </p>
                </div>
              )}
            </TabsContent>

            {/* Chat Tab */}
            <TabsContent value="chat" className="space-y-4 mt-0">
              <Card className="bg-workspace border-border">
                <CardHeader className="pb-3">
                  <CardTitle className="text-sm">Ask AI Assistant</CardTitle>
                  <CardDescription className="text-xs">
                    Ask questions about your documents
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-3">
                  <Textarea
                    placeholder="Ask a question about your documents..."
                    value={chatInput}
                    onChange={(e) => setChatInput(e.target.value)}
                    className="min-h-[100px] bg-panel border-border resize-none"
                    onKeyDown={(e) => {
                      if (e.key === "Enter" && !e.shiftKey) {
                        e.preventDefault();
                        handleAskQuestion();
                      }
                    }}
                  />
                  <Button
                    className="w-full"
                    size="sm"
                    onClick={handleAskQuestion}
                    disabled={!chatInput.trim() || !sessionId || isAsking}
                  >
                    {isAsking ? (
                      <div className="animate-spin rounded-full h-3 w-3 border-2 border-current border-t-transparent mr-1"></div>
                    ) : (
                      <MessageCircle className="h-3 w-3 mr-1" />
                    )}
                    {isAsking ? "Thinking..." : "Ask Question"}
                  </Button>
                </CardContent>
              </Card>

              {/* Chat History */}
              {chatHistory.length > 0 ? (
                <div className="space-y-3">
                  {chatHistory
                    .slice()
                    .reverse()
                    .map((chat) => (
                      <div key={chat.id} className="space-y-2">
                        {/* User Question */}
                        <Card className="bg-primary/10 border-primary/20">
                          <CardContent className="p-3">
                            <p className="text-sm font-medium mb-1">
                              You asked:
                            </p>
                            <p className="text-sm text-muted-foreground">
                              {chat.question}
                            </p>
                          </CardContent>
                        </Card>

                        {/* AI Answer */}
                        <Card className="bg-workspace border-border">
                          <CardContent className="p-3">
                            <div className="flex items-center gap-2 mb-2">
                              <Brain className="h-4 w-4" />
                              <p className="text-sm font-medium">
                                AI Assistant:
                              </p>
                            </div>
                            <p className="text-sm text-muted-foreground mb-3">
                              {chat.answer}
                            </p>

                            {/* Sources */}
                            {chat.sources && chat.sources.length > 0 && (
                              <div className="border-t border-border pt-2">
                                <p className="text-xs font-medium mb-2">
                                  Sources:
                                </p>
                                <div className="space-y-1">
                                  {chat.sources.map((source, idx) => (
                                    <div
                                      key={`${chat.id}-source-${idx}-${source.document}-${source.page_number}`}
                                      className="text-xs text-muted-foreground"
                                    >
                                      📄 {source.document} (Page{" "}
                                      {source.page_number})
                                    </div>
                                  ))}
                                </div>
                              </div>
                            )}
                          </CardContent>
                        </Card>
                      </div>
                    ))}
                </div>
              ) : (
                <div className="text-center text-muted-foreground">
                  <MessageCircle className="h-8 w-8 mx-auto mb-2 opacity-50" />
                  <p className="text-xs">
                    Start a conversation with your AI assistant
                  </p>
                </div>
              )}
            </TabsContent>
          </Tabs>
        </div>
      </ScrollArea>
    </div>
  );
}
