import { BrowserRouter, Routes, Route } from "react-router-dom";
import { useEffect } from "react";
import { useAppStore } from "./lib/store";
import SimplifiedUploadPage from "./pages/SimplifiedUploadPage";
import WorkspacePage from "./pages/WorkspacePage";
import KnowledgeGraphPage from "./pages/KnowledgeGraphPage";

function App() {
  const { currentView } = useAppStore();

  useEffect(() => {
    // Force dark mode for the research canvas theme
    document.documentElement.classList.add("dark");
  }, []);

  return (
    <div className="min-h-screen bg-background text-foreground">
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/workspace" element={<WorkspacePage />} />
          <Route path="/knowledge-graph" element={<KnowledgeGraphPage />} />
          <Route path="*" element={<HomePage />} />
        </Routes>
      </BrowserRouter>
    </div>
  );
}

function HomePage() {
  const { currentView, documents } = useAppStore();

  // If we have documents and current view is workspace, show workspace
  if (documents.length > 0 && currentView === "workspace") {
    return <WorkspacePage />;
  }

  // If current view is knowledge-graph, show knowledge graph
  if (currentView === "knowledge-graph") {
    return <KnowledgeGraphPage />;
  }

  // Default to upload page
  return <SimplifiedUploadPage />;
}

export default App;
