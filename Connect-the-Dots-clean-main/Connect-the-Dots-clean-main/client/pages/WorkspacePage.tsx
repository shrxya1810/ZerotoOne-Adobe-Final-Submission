import { useEffect, useRef, useState } from "react";
// Walkthrough steps definition
const WALKTHROUGH_STEPS = [
  {
    title: "Welcome to Your Research Canvas!",
    description:
      "This is your main workspace. Here you can view, highlight, and analyze your uploaded documents.",
    target: "workspace",
  },
  {
    title: "Sidebars",
    description:
      "Use the left and right sidebars to navigate documents, view insights, and manage your research.",
    target: "sidebars",
  },
  {
    title: "AI Insights",
    description:
      "Highlight text in your documents to generate AI-powered insights and connections.",
    target: "insights",
  },
];
import { useAppStore } from "../lib/store";
import LeftSidebar from "../components/workspace/LeftSidebar";
import DocumentViewer from "../components/workspace/DocumentViewer";
import RightSidebar from "../components/workspace/RightSidebar";
import WorkspaceHeader from "../components/workspace/WorkspaceHeader";
import PersonaPanel from "../components/workspace/PersonaPanel";
import { Button } from "../components/ui/button";
import {
  PanelLeftClose,
  PanelLeftOpen,
  PanelRightClose,
  PanelRightOpen,
} from "lucide-react";
import { cn } from "../lib/utils";
import gsap from "gsap";
import JourneyBar, { JourneyStep } from "../components/JourneyBar";
import ConnectingDotsBackground from "../components/ConnectingDotsBackground";

export default function WorkspacePage() {
  // Walkthrough state - check if user has already seen it
  const [walkthroughStep, setWalkthroughStep] = useState<number | null>(() => {
    const hasSeenWalkthrough = localStorage.getItem(
      "hasSeenWorkspaceWalkthrough",
    );
    return hasSeenWalkthrough ? null : 0;
  });

  const handleNextWalkthrough = () => {
    if (
      walkthroughStep !== null &&
      walkthroughStep < WALKTHROUGH_STEPS.length - 1
    ) {
      setWalkthroughStep(walkthroughStep + 1);
    } else {
      setWalkthroughStep(null);
      localStorage.setItem("hasSeenWorkspaceWalkthrough", "true");
    }
  };
  const handleSkipWalkthrough = () => {
    setWalkthroughStep(null);
    localStorage.setItem("hasSeenWorkspaceWalkthrough", "true");
  };
  // Always call useAppStore once, at the top, and destructure all needed values
  const store = useAppStore();
  const {
    leftSidebarOpen,
    rightSidebarOpen,
    toggleLeftSidebar,
    toggleRightSidebar,
    currentDocument,
    documents,
    currentStep,
  } = store;

  const workspaceRef = useRef<HTMLDivElement>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // Loading simulation and entrance animation
    const timer = setTimeout(() => {
      setIsLoading(false);

      if (workspaceRef.current) {
        gsap.fromTo(
          workspaceRef.current,
          { opacity: 0, scale: 0.95 },
          { opacity: 1, scale: 1, duration: 0.6, ease: "power2.out" },
        );
      }
    }, 500);

    return () => clearTimeout(timer);
  }, []);

  // Redirect to upload if no documents
  useEffect(() => {
    if (documents.length === 0) {
      // Could redirect to upload page
    }
  }, [documents]);

  if (isLoading) {
    return (
      <div className="min-h-screen bg-workspace flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-4 border-primary border-t-transparent mx-auto mb-4"></div>
          <p className="text-lg text-muted-foreground">
            Preparing your research canvas...
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-slate-900 text-white flex flex-col relative">
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
        <JourneyBar currentStep={Math.min(currentStep, 3) as JourneyStep} />
      </div>
      {/* Connecting Dots Background */}
      <ConnectingDotsBackground />
      {/* Header */}
      <WorkspaceHeader />

      {/* Main Workspace */}
      <div
        ref={workspaceRef}
        className={cn(
          "flex-1 flex relative z-10 h-[calc(100vh-5rem)]",
          walkthroughStep !== null &&
            WALKTHROUGH_STEPS[walkthroughStep].target === "workspace"
            ? "ring-4 ring-red-400 ring-offset-4 ring-offset-slate-900"
            : "",
        )}
      >
        {/* Left Sidebar */}
        <div
          className={cn(
            "transition-all duration-300 ease-in-out glass-panel border-r border-white/10",
            leftSidebarOpen ? "w-80" : "w-0",
            walkthroughStep !== null &&
              WALKTHROUGH_STEPS[walkthroughStep].target === "sidebars"
              ? "ring-4 ring-red-400 ring-offset-2 ring-offset-slate-900"
              : "",
          )}
        >
          {leftSidebarOpen && <LeftSidebar />}
        </div>

        {/* Left Sidebar Toggle */}
        <div className="flex flex-col justify-center">
          <Button
            variant="ghost"
            size="sm"
            onClick={toggleLeftSidebar}
            className="rounded-none border-y border-r border-white/10 glass hover:glass-button"
          >
            {leftSidebarOpen ? (
              <PanelLeftClose className="h-4 w-4" />
            ) : (
              <PanelLeftOpen className="h-4 w-4" />
            )}
          </Button>
        </div>

        {/* Center Panel - Document Viewer */}
        <div
          className={cn(
            "flex-1 flex flex-col min-w-0 h-full max-h-full overflow-hidden",
            walkthroughStep !== null &&
              WALKTHROUGH_STEPS[walkthroughStep].target === "insights"
              ? "ring-4 ring-red-400 ring-offset-2 ring-offset-slate-900"
              : "",
          )}
        >
          <div className="flex-1 min-h-0">
            <DocumentViewer />
          </div>
          <div className="flex-shrink-0">
            <PersonaPanel />
          </div>
        </div>

        {/* Right Sidebar Toggle */}
        <div className="flex flex-col justify-center">
          <Button
            variant="ghost"
            size="sm"
            onClick={toggleRightSidebar}
            className="rounded-none border-y border-l border-white/10 glass hover:glass-button"
          >
            {rightSidebarOpen ? (
              <PanelRightClose className="h-4 w-4" />
            ) : (
              <PanelRightOpen className="h-4 w-4" />
            )}
          </Button>
        </div>

        {/* Right Sidebar */}
        <div
          className={cn(
            "transition-all duration-300 ease-in-out glass-panel border-l border-white/10",
            rightSidebarOpen ? "w-80" : "w-0",
          )}
        >
          {rightSidebarOpen && <RightSidebar />}
        </div>
      </div>

      {/* Connection Thread Overlay for AI insights */}
      <div
        id="connection-thread"
        className="pointer-events-none fixed inset-0 z-50"
      >
        {/* GSAP will animate connection lines here */}
      </div>
    </div>
  );
}
