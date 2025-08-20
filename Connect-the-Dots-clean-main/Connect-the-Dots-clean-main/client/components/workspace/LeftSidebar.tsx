import { useState, useEffect } from "react";
import { useAppStore, OutlineType } from "../../lib/store";
import { BookOpen, Plus, FileText, File, Globe } from "lucide-react";
import { cn } from "../../lib/utils";
import { Button } from "../ui/button";

import { Badge } from "../ui/badge";

export default function LeftSidebar() {
  const {
    documents,
    currentDocument,
    setCurrentDocument,
    setCurrentView,
    outlines,
    activeOutlineSection,
    setActiveOutlineSection,
  } = useAppStore();
  const [activeTab, setActiveTab] = useState<"documents" | "outline">(
    "documents",
  );
  const addMoreDocuments = () => setCurrentView("upload");
  // Remove current document from workspace and ready for analysis
  const removeCurrentDocument = () => {
    if (currentDocument) {
      // Remove from zustand store
      const { removeDocument, setCurrentDocument } = useAppStore.getState();
      removeDocument(currentDocument.id);
      setCurrentDocument(null);
    }
  };
  const getFileIcon = (type: string) => {
    switch (type) {
      case "pdf":
        return <FileText className="h-4 w-4 text-red-400" />;
      case "docx":
        return <File className="h-4 w-4 text-blue-400" />;
      case "txt":
        return <FileText className="h-4 w-4 text-gray-400" />;
      case "web":
        return <Globe className="h-4 w-4 text-green-400" />;
      default:
        return <File className="h-4 w-4" />;
    }
  };
  const outlineItems: OutlineType = Array.isArray(
    currentDocument && outlines[currentDocument?.id],
  )
    ? outlines[currentDocument.id]
    : [];

  // Debug: log outline items when they change
  useEffect(() => {
    if (currentDocument && outlineItems.length > 0) {
      console.log("LeftSidebar: Current outline items:", outlineItems);
      console.log(
        "LeftSidebar: Sample outline item structure:",
        outlineItems[0],
      );
    }
  }, [currentDocument, outlineItems]);

  // Outline navigation handler
  const jumpToSection = (item: any) => {
    console.log("LeftSidebar: jumpToSection called with item:", item);
    if (item.page !== undefined) {
      // Set active section
      if (currentDocument) {
        const sectionId = item.id || `${item.text}-${item.page}`;
        setActiveOutlineSection(currentDocument.id, sectionId);
      }
      
      // Convert 0-indexed page number to 1-based for Adobe viewer
      const adobePageNumber = item.page + 1;
      console.log(
        `LeftSidebar: Converting page ${item.page} (0-indexed) to ${adobePageNumber} (1-based) for Adobe`,
      );
      console.log(
        `LeftSidebar: Dispatching go-to-page event for page ${adobePageNumber} with text "${item.text}"`,
      );
      window.dispatchEvent(
        new CustomEvent("go-to-page", { 
          detail: { 
            page: adobePageNumber,
            text: item.text // Pass the heading text for highlighting
          } 
        }),
      );
    } else {
      console.warn("LeftSidebar: No page number found in outline item:", item);
    }
  };

  return (
    <div className="h-full flex flex-col">
      {/* Tab Switcher (optional, if you want to switch between documents/outline) */}
      <div className="flex border-b">
        <Button
          variant={activeTab === "documents" ? "default" : "ghost"}
          className="flex-1 rounded-none"
          onClick={() => setActiveTab("documents")}
        >
          Documents
        </Button>
        <Button
          variant={activeTab === "outline" ? "default" : "ghost"}
          className="flex-1 rounded-none"
          onClick={() => setActiveTab("outline")}
        >
          Outline
        </Button>
      </div>
      <div className="flex-1 overflow-y-auto">
        {activeTab === "documents" ? (
          <div className="p-4 pb-6">
            {/* Session Collection */}
            <div className="mb-4">
              {documents.map((doc) => (
                <Button
                  key={doc.id}
                  variant={currentDocument?.id === doc.id ? "default" : "ghost"}
                  onClick={() => setCurrentDocument(doc)}
                  className={cn(
                    "w-full justify-start p-2 h-auto text-sm",
                    currentDocument?.id === doc.id &&
                      "bg-primary text-primary-foreground",
                  )}
                >
                  <div className="flex items-start gap-2 w-full">
                    {getFileIcon(doc.type)}
                    <div className="flex-1 text-left min-w-0">
                      <p className="font-medium truncate">{doc.name}</p>
                      <p className="text-xs opacity-70">
                        {(doc.size / 1024 / 1024).toFixed(1)} MB
                      </p>
                    </div>
                    {doc.isCurrentReading && (
                      <Badge variant="outline" className="text-xs">
                        Reading
                      </Badge>
                    )}
                  </div>
                </Button>
              ))}
              <Button
                variant="outline"
                onClick={addMoreDocuments}
                className="w-full justify-start p-2 h-auto text-sm border-dashed"
              >
                <Plus className="h-4 w-4 mr-2" />
                Add Documents
              </Button>
              <Button
                variant="outline"
                onClick={removeCurrentDocument}
                className="w-full justify-start p-2 h-auto text-sm border-dashed mt-2 text-red-400 border-red-400 hover:bg-red-900/20"
                disabled={!currentDocument}
              >
                Remove Document
              </Button>
            </div>
            {/* Document Stats */}
            <div className="bg-workspace rounded-lg p-3 space-y-2">
              <h3 className="text-sm font-medium">Session Overview</h3>
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div>
                  <span className="text-muted-foreground">Documents:</span>
                  <span className="ml-1 font-medium">{documents.length}</span>
                </div>
                <div>
                  <span className="text-muted-foreground">Total Size:</span>
                  <span className="ml-1 font-medium">
                    {(
                      documents.reduce((acc, doc) => acc + doc.size, 0) /
                      1024 /
                      1024
                    ).toFixed(1)}{" "}
                    MB
                  </span>
                </div>
              </div>
            </div>
          </div>
        ) : (
          <div className="p-4 pb-6">
            {currentDocument ? (
              <>
                <div className="flex items-center gap-2 mb-4">
                  {getFileIcon(currentDocument.type)}
                  <h3 className="font-medium text-sm truncate">
                    {currentDocument.name}
                  </h3>
                </div>
                <div className="space-y-1">
                  {outlineItems && outlineItems.length > 0 ? (
                    outlineItems.map((item, idx) => {
                      const sectionId = item.id || `${item.text}-${item.page}`;
                      const isActive = currentDocument && activeOutlineSection[currentDocument.id] === sectionId;
                      
                      return (
                        <Button
                          key={item.id || idx}
                          variant={isActive ? "secondary" : "ghost"}
                          className={`w-full justify-start text-xs px-2 py-1 text-left transition-colors ${
                            isActive 
                              ? "bg-primary/10 text-primary border-l-2 border-primary font-medium" 
                              : "hover:bg-muted"
                          }`}
                          onClick={() => jumpToSection(item)}
                        >
                          {item.text}{" "}
                          {item.page !== undefined ? (
                            <span className="ml-2 text-muted-foreground">
                              (p.{item.page + 1})
                            </span>
                          ) : null}
                        </Button>
                      );
                    })
                  ) : (
                    <div className="text-xs text-muted-foreground">
                      No outline available for this document.
                    </div>
                  )}
                </div>
              </>
            ) : (
              <div className="text-center text-muted-foreground text-sm">
                <BookOpen className="h-8 w-8 mx-auto mb-2 opacity-50" />
                <p>Select a document to view its outline</p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
  // ...existing code ends here
}
