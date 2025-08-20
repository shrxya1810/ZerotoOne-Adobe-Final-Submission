import { useCallback, useState, useRef, useEffect } from "react";
// Walkthrough steps definition

import JourneyBar from "../components/JourneyBar";
import { useDropzone } from "react-dropzone";
import { useAppStore, Document } from "../lib/store";
import { processPdf, createSession, uploadDocuments } from "../../shared/api";
import { Button } from "../components/ui/button";
import { Input } from "../components/ui/input";
import { Label } from "../components/ui/label";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "../components/ui/card";
import { Badge } from "../components/ui/badge";
import {
  Upload,
  FileText,
  File,
  Globe,
  X,
  Play,
  Brain,
  Network,
  Zap,
  ArrowDown,
  Volume2,
  Search as SearchIcon,
  Sparkles,
} from "lucide-react";
import { cn } from "../lib/utils";
import gsap from "gsap";
import ConnectingDotsBackground from "../components/ConnectingDotsBackground";

const WALKTHROUGH_STEPS = [
  {
    title: "Welcome to Connect the Dots!",
    description:
      "This quick walkthrough will show you how to get started with AI-powered document assistant. Click Next to continue.",
    target: "hero",
  },
  {
    title: "Upload Your Documents",
    description:
      "Upload PDFs here. You can add multiple documents for cross-document insights.",
    target: "upload",
  },
  {
    title: "Launch Research Canvas",
    description:
      "Once your documents are uploaded, click here to launch your research workspace and start exploring.",
    target: "launch",
  },
];

export default function SimplifiedUploadPage() {
  // Walkthrough state
  const [walkthroughStep, setWalkthroughStep] = useState<number | null>(0); // null = not showing

  const handleNextWalkthrough = () => {
    if (
      walkthroughStep !== null &&
      walkthroughStep < WALKTHROUGH_STEPS.length - 1
    ) {
      setWalkthroughStep(walkthroughStep + 1);
    } else {
      setWalkthroughStep(null);
    }
  };
  const handleSkipWalkthrough = () => setWalkthroughStep(null);
  const {
    documents,
    addDocuments,
    setCurrentView,
    sessionId,
    sessionName,
    setSessionId,
    setSessionName,
    isUploading,
    setIsUploading,
    currentStep,
    setCurrentStep,
  } = useAppStore();

  const [dragActive, setDragActive] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState<File[]>([]);
  const [isScrolled, setIsScrolled] = useState(false);

  const heroRef = useRef<HTMLDivElement>(null);
  const uploadSectionRef = useRef<HTMLDivElement>(null);
  const featuresRef = useRef<HTMLDivElement>(null);
  const titleRef = useRef<HTMLDivElement>(null);
  const subtitleRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Set progress bar based on document status
    if (documents.length === 0) {
      setCurrentStep(0);
    } else {
      setCurrentStep(1);
    }
  }, [documents, setCurrentStep]);
  useEffect(() => {
    // Initial hero animation
    const tl = gsap.timeline();
    tl.fromTo(
      titleRef.current,
      { opacity: 0, y: 100, scale: 0.8 },
      { opacity: 1, y: 0, scale: 1, duration: 1.5, ease: "power3.out" },
    ).fromTo(
      subtitleRef.current,
      { opacity: 0, y: 50 },
      { opacity: 1, y: 0, duration: 1, ease: "power2.out" },
      "-=0.8",
    );

    // Scroll handler
    const handleScroll = () => {
      const scrollY = window.scrollY;

      if (scrollY > 100 && !isScrolled) {
        setIsScrolled(true);

        // Animate title when scrolling
        gsap.to(titleRef.current, {
          y: -100,
          scale: 0.8,
          opacity: 0.7,
          duration: 0.6,
          ease: "power2.inOut",
        });

        gsap.to(subtitleRef.current, {
          y: -80,
          opacity: 0.5,
          duration: 0.6,
          ease: "power2.inOut",
        });
      } else if (scrollY <= 100 && isScrolled) {
        setIsScrolled(false);

        // Reset title position
        gsap.to(titleRef.current, {
          y: 0,
          scale: 1,
          opacity: 1,
          duration: 0.6,
          ease: "power2.inOut",
        });

        gsap.to(subtitleRef.current, {
          y: 0,
          opacity: 1,
          duration: 0.6,
          ease: "power2.inOut",
        });
      }
    };

    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, [isScrolled]);

  const { setOutline } = useAppStore();

  const onDrop = useCallback(
    async (acceptedFiles: File[]) => {
      setUploadedFiles((prev) => [...prev, ...acceptedFiles]);
      setIsUploading(true);

      try {
        // Create session if one doesn't exist
        let currentSessionId = sessionId;
        if (!currentSessionId) {
          const sessionResponse = await createSession(sessionName || undefined);
          currentSessionId = sessionResponse.session.session_id;
          setSessionId(currentSessionId);
        }

        // Upload documents to the session
        const uploadResponse = await uploadDocuments(
          currentSessionId,
          acceptedFiles,
        );

        // Create document objects for the frontend store
        const newDocuments: Document[] = acceptedFiles.map((file) => ({
          id: Math.random().toString(36).substr(2, 9),
          name: file.name,
          type: getFileType(file.name),
          size: file.size,
          uploadedAt: new Date(),
          url: `/api/pdfs/${currentSessionId}/${encodeURIComponent(file.name)}`,
        }));

        // Process PDFs for outline extraction
        for (let i = 0; i < acceptedFiles.length; i++) {
          const file = acceptedFiles[i];
          const doc = newDocuments[i];

          if (doc.type === "pdf") {
            try {
              const outline = await processPdf(file);
              setOutline(doc.id, outline);
            } catch (e) {
              console.error("Failed to extract outline:", e);
              setOutline(doc.id, []);
            }
          }
        }

        addDocuments(newDocuments);
      } catch (error) {
        console.error("Upload failed:", error);
        // Handle error - maybe show a toast notification
      } finally {
        setIsUploading(false);
      }
    },
    [
      sessionId,
      sessionName,
      setSessionId,
      addDocuments,
      setIsUploading,
      setOutline,
    ],
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      "application/pdf": [".pdf"],
      "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        [".docx"],
      "text/plain": [".txt"],
      "text/markdown": [".md"],
    },
    multiple: true,
  });

  const getFileType = (filename: string): Document["type"] => {
    const ext = filename.split(".").pop()?.toLowerCase();
    switch (ext) {
      case "pdf":
        return "pdf";
      case "docx":
        return "docx";
      case "txt":
      case "md":
        return "txt";
      default:
        return "txt";
    }
  };

  const getFileIcon = (type: Document["type"]) => {
    switch (type) {
      case "pdf":
        return <FileText className="h-4 w-4" />;
      case "docx":
        return <File className="h-4 w-4" />;
      case "txt":
        return <FileText className="h-4 w-4" />;
      case "web":
        return <Globe className="h-4 w-4" />;
      default:
        return <File className="h-4 w-4" />;
    }
  };

  // Remove file from uploadedFiles and also from ready for analysis (documents)
  const removeFile = (index: number) => {
    setUploadedFiles((prev) => prev.filter((_, i) => i !== index));
    // Remove from ready for analysis if it was already processed
    const fileToRemove = uploadedFiles[index];
    if (fileToRemove) {
      // Remove from zustand store if a Document with same name exists
      const { documents, removeDocument } = useAppStore.getState();
      const doc = documents.find(
        (d) => d.name === fileToRemove.name && d.size === fileToRemove.size,
      );
      if (doc) {
        removeDocument(doc.id);
      }
    }
  };

  const startResearchSession = () => {
    if (documents.length > 0) {
      setCurrentView("workspace");
    }
  };

  const scrollToUpload = () => {
    uploadSectionRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  return (
    <div className="bg-slate-900 text-white min-h-screen">
      {/* Walkthrough Modal */}
      {walkthroughStep !== null && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/60">
          <div className="bg-slate-800 rounded-2xl shadow-2xl p-8 max-w-md w-full text-center border border-red-400 relative animate-fade-in">
            <h3 className="text-2xl font-bold mb-4 text-white">
              {WALKTHROUGH_STEPS[walkthroughStep].title}
            </h3>
            <p className="text-slate-300 mb-6 text-base">
              {WALKTHROUGH_STEPS[walkthroughStep].description}
            </p>
            <div className="flex gap-4 justify-center">
              <Button
                onClick={handleSkipWalkthrough}
                variant="ghost"
                className="text-slate-400 hover:text-red-400"
              >
                Skip
              </Button>
              <Button
                onClick={handleNextWalkthrough}
                className="glass-button text-white font-semibold px-6 py-2 rounded-lg"
              >
                {walkthroughStep === WALKTHROUGH_STEPS.length - 1
                  ? "Finish"
                  : "Next"}
              </Button>
            </div>
            <div className="absolute top-2 right-4 text-xs text-slate-400">
              {walkthroughStep + 1}/{WALKTHROUGH_STEPS.length}
            </div>
          </div>
        </div>
      )}
      <div className="sticky top-0 z-50 bg-slate-900/80 backdrop-blur-md shadow-lg border-b border-red-400">
        <JourneyBar currentStep={currentStep} />
      </div>
      {/* Connecting Dots Background */}
      <ConnectingDotsBackground />

      {/* Hero Section */}
      <section
        ref={heroRef}
        className={cn(
          "relative z-10 min-h-screen flex flex-col items-center justify-center px-6",
          walkthroughStep !== null &&
            WALKTHROUGH_STEPS[walkthroughStep].target === "hero"
            ? "ring-4 ring-red-400 ring-offset-4 ring-offset-slate-900"
            : "",
        )}
      >
        <div className="text-center max-w-5xl mx-auto">
          <div ref={titleRef} className="mb-8">
            <h1 className="text-7xl md:text-8xl lg:text-9xl font-display font-black mb-4 leading-none">
              <span className="block bg-gradient-to-r from-white via-red-200 to-red-400 bg-clip-text text-transparent glow-text">
                Connecting
              </span>
              <span className="block bg-gradient-to-r from-red-400 via-red-500 to-white bg-clip-text text-transparent glow-text">
                the Dots
              </span>
            </h1>
          </div>

          <div ref={subtitleRef} className="mb-12">
            <p className="text-xl md:text-2xl text-slate-300 mb-6 max-w-3xl mx-auto leading-relaxed">
              Transform your research with AI-powered insights that reveal
              hidden connections across your documents. Experience the future of
              knowledge discovery.
            </p>

            <div className="flex flex-wrap items-center justify-center gap-6 text-slate-400">
              <div className="flex items-center gap-2">
                <Brain className="h-5 w-5 text-red-400" />
                <span>AI Insights</span>
              </div>
              <div className="flex items-center gap-2">
                <Network className="h-5 w-5 text-red-400" />
                <span>Knowledge Graphs</span>
              </div>
              <div className="flex items-center gap-2">
                <Volume2 className="h-5 w-5 text-red-400" />
                <span>Audio Podcasts</span>
              </div>
            </div>
          </div>

          <Button
            onClick={scrollToUpload}
            className="glass-button text-white font-semibold px-8 py-4 rounded-full hover:scale-105 transition-all duration-300 pulse-glow"
          >
            <Sparkles className="h-5 w-5 mr-2" />
            Start Exploring
            <ArrowDown className="h-5 w-5 ml-2" />
          </Button>
        </div>
      </section>

      {/* Upload Section */}
      <section
        ref={uploadSectionRef}
        className={cn(
          "relative z-10 min-h-screen flex items-center justify-center px-6 py-20",
          walkthroughStep !== null &&
            WALKTHROUGH_STEPS[walkthroughStep].target === "upload"
            ? "ring-4 ring-red-400 ring-offset-4 ring-offset-slate-900"
            : "",
        )}
      >
        <div className="max-w-4xl mx-auto w-full">
          {/* Session Name Input */}
          <div className="text-center mb-12">
            <h2 className="text-4xl font-display font-bold mb-6 bg-gradient-to-r from-white to-red-300 bg-clip-text text-transparent">
              Begin Your Journey
            </h2>
            <div className="max-w-md mx-auto">
              <Label
                htmlFor="session-name"
                className="text-sm font-medium mb-2 block text-slate-300"
              >
                Session Name (Optional)
              </Label>
              <Input
                id="session-name"
                placeholder="e.g., Renewable Energy Research"
                value={sessionName}
                onChange={(e) => setSessionName(e.target.value)}
                className="glass text-white placeholder:text-slate-400"
              />
            </div>
          </div>

          {/* Upload Area */}
          <Card className="glass-card border-0 shadow-2xl">
            <CardHeader className="text-center">
              <CardTitle className="flex items-center justify-center gap-2 text-2xl font-display text-white">
                <Upload className="h-6 w-6 text-red-400" />
                Upload Documents
              </CardTitle>
              <CardDescription className="text-slate-300 text-lg">
                Upload PDF multiple documents at once.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div
                {...getRootProps()}
                className={cn(
                  "border-2 border-dashed rounded-xl p-16 text-center cursor-pointer transition-all duration-500 relative overflow-hidden",
                  isDragActive || dragActive
                    ? "border-red-400 glass-button glow-red"
                    : "border-slate-600 hover:border-red-400 hover:glass-button",
                )}
              >
                <input {...getInputProps()} />
                <div className="relative z-10">
                  <Upload className="h-20 w-20 mx-auto mb-6 text-slate-400" />
                  <h3 className="text-2xl font-bold mb-4 text-white">
                    {isDragActive ? "Drop files here" : "Drag & drop documents"}
                  </h3>
                  <p className="text-slate-300 mb-6 text-lg">
                    or click to browse files
                  </p>
                  <Button className="glass-button text-white font-semibold px-6 py-3 rounded-lg">
                    Choose Files
                  </Button>
                </div>
              </div>

              {/* Uploaded Files */}
              {uploadedFiles.length > 0 && (
                <div className="mt-8">
                  <h4 className="font-medium mb-4 text-white text-lg">
                    Uploaded Files ({uploadedFiles.length})
                  </h4>
                  <div className="grid gap-3">
                    {uploadedFiles.map((file, index) => (
                      <div
                        key={index}
                        className="flex items-center justify-between p-4 glass rounded-lg border border-slate-600"
                      >
                        <div className="flex items-center gap-3">
                          {getFileIcon(getFileType(file.name))}
                          <div>
                            <p className="font-medium text-white">
                              {file.name}
                            </p>
                            <p className="text-sm text-slate-400">
                              {(file.size / 1024 / 1024).toFixed(2)} MB
                            </p>
                          </div>
                        </div>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => removeFile(index)}
                          className="text-slate-400 hover:text-red-400"
                        >
                          <X className="h-4 w-4" />
                        </Button>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Processed Documents */}
              {documents.length > 0 && (
                <div className="mt-8">
                  <div className="flex items-center justify-between mb-4">
                    <h4 className="font-medium text-white text-lg">
                      Ready for Analysis ({documents.length})
                    </h4>
                    <Badge className="glass-button text-red-400 border-red-400">
                      Processed
                    </Badge>
                  </div>
                  <div className="grid gap-3 max-h-40 overflow-y-auto">
                    {documents.map((doc) => (
                      <div
                        key={doc.id}
                        className="flex items-center gap-3 p-3 glass rounded border border-slate-600"
                      >
                        {getFileIcon(doc.type)}
                        <span className="font-medium text-white">
                          {doc.name}
                        </span>
                      </div>
                    ))}
                  </div>

                  <Button
                    onClick={startResearchSession}
                    className={cn(
                      "w-full mt-6 glass-button text-white hover:scale-[1.02] transition-all duration-300 py-4 text-lg font-semibold glow-red",
                      walkthroughStep !== null &&
                        WALKTHROUGH_STEPS[walkthroughStep].target === "launch"
                        ? "ring-4 ring-red-400 ring-offset-2 ring-offset-slate-900"
                        : "",
                    )}
                    size="lg"
                  >
                    <Play className="h-5 w-5 mr-2" />
                    Launch Research Canvas
                  </Button>
                </div>
              )}

              {isUploading && (
                <div className="mt-8 text-center">
                  <div className="inline-flex items-center gap-3 text-red-400 text-lg">
                    <div className="animate-spin rounded-full h-6 w-6 border-2 border-red-400 border-t-transparent"></div>
                    Processing documents with AI...
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </section>

      {/* Features Section */}
      <section ref={featuresRef} className="relative z-10 py-20 px-6">
        <div className="max-w-7xl mx-auto">
          <h2 className="text-5xl font-display font-bold text-center mb-16 bg-gradient-to-r from-white to-red-300 bg-clip-text text-transparent">
            AI-Powered Research Intelligence
          </h2>

          <div className="grid md:grid-cols-3 gap-8">
            <Card className="glass-card border-0 shadow-xl hover:scale-[1.02] transition-all duration-300">
              <CardHeader className="text-center">
                {/* AI Insight Generation Animation */}
                <div className="w-20 h-20 mx-auto mb-6 relative">
                  {/* Neural network visualization */}
                  <div className="absolute inset-0 rounded-lg">
                    {/* Nodes */}
                    <div className="absolute top-2 left-3 w-2 h-2 bg-yellow-400 rounded-full animate-pulse"></div>
                    <div
                      className="absolute top-6 right-4 w-2 h-2 bg-yellow-300 rounded-full animate-pulse"
                      style={{ animationDelay: "0.5s" }}
                    ></div>
                    <div
                      className="absolute bottom-3 left-5 w-2 h-2 bg-yellow-500 rounded-full animate-pulse"
                      style={{ animationDelay: "1s" }}
                    ></div>
                    <div
                      className="absolute bottom-5 right-2 w-2 h-2 bg-yellow-200 rounded-full animate-pulse"
                      style={{ animationDelay: "1.5s" }}
                    ></div>

                    {/* Connecting lines */}
                    <svg className="absolute inset-0 w-full h-full">
                      <line
                        x1="16"
                        y1="12"
                        x2="60"
                        y2="28"
                        stroke="rgb(255,215,0)"
                        strokeWidth="1"
                        opacity="0.4"
                      >
                        <animate
                          attributeName="opacity"
                          values="0.2;0.8;0.2"
                          dur="2s"
                          repeatCount="indefinite"
                        />
                      </line>
                      <line
                        x1="16"
                        y1="12"
                        x2="24"
                        y2="60"
                        stroke="rgb(255,215,0)"
                        strokeWidth="1"
                        opacity="0.4"
                      >
                        <animate
                          attributeName="opacity"
                          values="0.2;0.8;0.2"
                          dur="2.5s"
                          repeatCount="indefinite"
                        />
                      </line>
                      <line
                        x1="60"
                        y1="28"
                        x2="68"
                        y2="52"
                        stroke="rgb(255,215,0)"
                        strokeWidth="1"
                        opacity="0.4"
                      >
                        <animate
                          attributeName="opacity"
                          values="0.2;0.8;0.2"
                          dur="1.8s"
                          repeatCount="indefinite"
                        />
                      </line>
                    </svg>

                    {/* Central brain icon */}
                    <div className="absolute inset-0 flex items-center justify-center">
                      <Brain className="h-6 w-6 text-yellow-400 animate-pulse" />
                    </div>

                    {/* Insight bubbles */}
                    <div
                      className="absolute top-1 right-1 w-3 h-3 bg-yellow-300 rounded-full opacity-70 animate-ping"
                      style={{ animationDelay: "0.5s" }}
                    ></div>
                    <div
                      className="absolute bottom-1 left-1 w-2 h-2 bg-yellow-400 rounded-full opacity-70 animate-ping"
                      style={{ animationDelay: "1.2s" }}
                    ></div>
                  </div>
                </div>
                <CardTitle className="text-xl text-white font-display">
                  AI-Generated Insights
                </CardTitle>
                <CardDescription className="text-slate-300 text-base">
                  Get instant contradictions, enhancements, and connections as
                  you select text in your documents.
                </CardDescription>
              </CardHeader>
            </Card>

            <Card className="glass-card border-0 shadow-xl hover:scale-[1.02] transition-all duration-300">
              <CardHeader className="text-center">
                {/* Text Analysis Animation */}
                <div className="w-20 h-20 mx-auto mb-6 relative">
                  {/* Document with text lines */}
                  <div className="absolute inset-2 bg-gray-800 rounded border border-yellow-400/30">
                    {/* Text lines */}
                    <div className="p-2 space-y-1">
                      <div
                        className="h-1 bg-gray-400 rounded"
                        style={{ width: "80%" }}
                      >
                        <div
                          className="h-full bg-yellow-400 rounded animate-pulse"
                          style={{ width: "30%" }}
                        ></div>
                      </div>
                      <div
                        className="h-1 bg-gray-400 rounded"
                        style={{ width: "90%" }}
                      >
                        <div
                          className="h-full bg-yellow-400 rounded animate-pulse"
                          style={{ width: "60%", animationDelay: "0.5s" }}
                        ></div>
                      </div>
                      <div
                        className="h-1 bg-gray-400 rounded"
                        style={{ width: "75%" }}
                      >
                        <div
                          className="h-full bg-yellow-400 rounded animate-pulse"
                          style={{ width: "40%", animationDelay: "1s" }}
                        ></div>
                      </div>
                      <div
                        className="h-1 bg-gray-400 rounded"
                        style={{ width: "85%" }}
                      >
                        <div
                          className="h-full bg-yellow-400 rounded animate-pulse"
                          style={{ width: "70%", animationDelay: "1.5s" }}
                        ></div>
                      </div>
                    </div>
                  </div>

                  {/* Scanning beam */}
                  <div className="absolute inset-0 overflow-hidden rounded">
                    <div className="absolute top-0 left-0 w-full h-0.5 bg-gradient-to-r from-transparent via-yellow-400 to-transparent animate-pulse">
                      <div className="w-full h-full animate-pulse"></div>
                    </div>
                  </div>

                  {/* Search icon */}
                  <div className="absolute -top-1 -right-1 bg-yellow-400 rounded-full p-1">
                    <SearchIcon className="h-3 w-3 text-black" />
                  </div>

                  {/* Analysis particles */}
                  <div
                    className="absolute top-0 left-0 w-1 h-1 bg-yellow-300 rounded-full animate-ping"
                    style={{ animationDelay: "0.3s" }}
                  ></div>
                  <div
                    className="absolute bottom-2 right-2 w-1 h-1 bg-yellow-400 rounded-full animate-ping"
                    style={{ animationDelay: "0.8s" }}
                  ></div>
                </div>
                <CardTitle className="text-xl text-white font-display">
                  Relevant Section Extraction
                </CardTitle>
                <CardDescription className="text-slate-300 text-base">
                  Automatically surface related passages from all your documents
                  based on your current selection.
                </CardDescription>
              </CardHeader>
            </Card>

            <Card className="glass-card border-0 shadow-xl hover:scale-[1.02] transition-all duration-300">
              <CardHeader className="text-center">
                {/* Knowledge Graph Building Animation */}
                <div className="w-20 h-20 mx-auto mb-6 relative">
                  {/* Graph nodes appearing progressively */}
                  <div className="absolute top-2 left-3 w-3 h-3 bg-yellow-400 rounded-full animate-pulse"></div>
                  <div
                    className="absolute top-8 right-2 w-3 h-3 bg-yellow-300 rounded-full animate-pulse"
                    style={{ animationDelay: "1s" }}
                  ></div>
                  <div
                    className="absolute bottom-8 left-6 w-3 h-3 bg-yellow-500 rounded-full animate-pulse"
                    style={{ animationDelay: "2s" }}
                  ></div>
                  <div
                    className="absolute bottom-2 right-6 w-3 h-3 bg-yellow-200 rounded-full animate-pulse"
                    style={{ animationDelay: "3s" }}
                  ></div>
                  <div
                    className="absolute top-6 left-12 w-3 h-3 bg-yellow-600 rounded-full animate-pulse"
                    style={{ animationDelay: "4s" }}
                  ></div>

                  {/* Connecting lines appearing */}
                  <svg className="absolute inset-0 w-full h-full">
                    <line
                      x1="16"
                      y1="14"
                      x2="64"
                      y2="38"
                      stroke="rgb(255,215,0)"
                      strokeWidth="1.5"
                      opacity="0.6"
                    >
                      <animate
                        attributeName="opacity"
                        values="0;0.8;0.8"
                        dur="5s"
                        repeatCount="indefinite"
                        begin="1s"
                      />
                      <animate
                        attributeName="stroke-dasharray"
                        values="0,100;50,0"
                        dur="1s"
                        begin="1s"
                      />
                    </line>
                    <line
                      x1="16"
                      y1="14"
                      x2="32"
                      y2="56"
                      stroke="rgb(255,215,0)"
                      strokeWidth="1.5"
                      opacity="0.6"
                    >
                      <animate
                        attributeName="opacity"
                        values="0;0.8;0.8"
                        dur="5s"
                        repeatCount="indefinite"
                        begin="2s"
                      />
                      <animate
                        attributeName="stroke-dasharray"
                        values="0,100;50,0"
                        dur="1s"
                        begin="2s"
                      />
                    </line>
                    <line
                      x1="64"
                      y1="38"
                      x2="56"
                      y2="68"
                      stroke="rgb(255,215,0)"
                      strokeWidth="1.5"
                      opacity="0.6"
                    >
                      <animate
                        attributeName="opacity"
                        values="0;0.8;0.8"
                        dur="5s"
                        repeatCount="indefinite"
                        begin="3s"
                      />
                      <animate
                        attributeName="stroke-dasharray"
                        values="0,100;50,0"
                        dur="1s"
                        begin="3s"
                      />
                    </line>
                    <line
                      x1="32"
                      y1="56"
                      x2="56"
                      y2="68"
                      stroke="rgb(255,215,0)"
                      strokeWidth="1.5"
                      opacity="0.6"
                    >
                      <animate
                        attributeName="opacity"
                        values="0;0.8;0.8"
                        dur="5s"
                        repeatCount="indefinite"
                        begin="4s"
                      />
                      <animate
                        attributeName="stroke-dasharray"
                        values="0,100;40,0"
                        dur="1s"
                        begin="4s"
                      />
                    </line>
                  </svg>

                  {/* Central network icon */}
                  <div className="absolute inset-0 flex items-center justify-center">
                    <Network
                      className="h-5 w-5 text-yellow-400 animate-pulse"
                      style={{ animationDelay: "2.5s" }}
                    />
                  </div>

                  {/* Building effect particles */}
                  <div
                    className="absolute top-0 left-0 w-1 h-1 bg-yellow-300 rounded-full animate-ping"
                    style={{ animationDelay: "1s" }}
                  ></div>
                  <div
                    className="absolute top-4 right-0 w-1 h-1 bg-yellow-400 rounded-full animate-ping"
                    style={{ animationDelay: "2s" }}
                  ></div>
                  <div
                    className="absolute bottom-0 left-4 w-1 h-1 bg-yellow-500 rounded-full animate-ping"
                    style={{ animationDelay: "3s" }}
                  ></div>
                </div>
                <CardTitle className="text-xl text-white font-display">
                  Knowledge Graph
                </CardTitle>
                <CardDescription className="text-slate-300 text-base">
                  Visualize connections between concepts, topics, and entities
                  across all your research documents.
                </CardDescription>
              </CardHeader>
            </Card>
          </div>
        </div>
      </section>
    </div>
  );
}
