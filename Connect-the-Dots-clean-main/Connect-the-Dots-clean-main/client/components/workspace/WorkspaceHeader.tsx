import { useState, useEffect, useRef } from "react";
import { useAppStore, SearchResult } from "../../lib/store";
import { semanticSearch, keywordSearch } from "../../../shared/api";
import { Button } from "../ui/button";
import { Input } from "../ui/input";
import { Badge } from "../ui/badge";
import {
  Search,
  Settings,
  Download,
  Brain,
  Network,
  Upload,
  Save,
  MoreHorizontal,
  ChevronRight,
  FileText,
  Loader2,
} from "lucide-react";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "../ui/dropdown-menu";
import { cn } from "../../lib/utils";

export default function WorkspaceHeader() {
  const {
    searchQuery,
    setSearchQuery,
    currentView,
    setCurrentView,
    sessionName,
    documents,
    isDirty,
    sessionId,
    searchResults,
    isSearching,
    showSearchDropdown,
    setShowSearchDropdown,
    performSearch,
    setCurrentDocument,
    currentDocument,
  } = useAppStore();

  const [searchMethod, setSearchMethod] = useState<"semantic" | "keyword">(
    "semantic",
  );
  const [actualSearchMethod, setActualSearchMethod] = useState<
    "semantic" | "keyword"
  >("semantic");

  const [searchFocused, setSearchFocused] = useState(false);
  const searchTimeoutRef = useRef<NodeJS.Timeout>();
  const searchContainerRef = useRef<HTMLDivElement>(null);

  // Custom search function that uses specific method
  const performSearchWithMethod = async (
    query: string,
    sessionId: string,
    method: "semantic" | "keyword",
  ) => {
    if (!query.trim() || !sessionId) return;

    const { setSearchResults, setIsSearching } = useAppStore.getState();

    try {
      setIsSearching(true);
      console.log(`🔍 Performing ${method} search...`, {
        query,
        sessionId,
        method,
      });

      let response;
      let actualMethod = method;

      try {
        if (method === "semantic") {
          response = await semanticSearch(sessionId, query, 10);
        } else {
          response = await keywordSearch(sessionId, query, 10);
        }
        console.log(`✅ ${method} search succeeded`);
      } catch (primaryError) {
        console.error(`❌ ${method} search failed with error:`, {
          error: primaryError,
          message: primaryError.message,
          stack: primaryError.stack,
        });

        // Only fallback if the primary method fails
        if (method === "keyword") {
          try {
            console.log(
              "🔄 Keyword search failed, trying semantic fallback...",
            );
            response = await semanticSearch(sessionId, query, 10);
            actualMethod = "semantic";
            console.log("✅ Semantic fallback successful");
          } catch (fallbackError) {
            console.error("❌ Semantic fallback also failed:", fallbackError);
            throw fallbackError;
          }
        } else {
          // If semantic fails, try keyword as fallback
          try {
            console.log(
              "🔄 Semantic search failed, trying keyword fallback...",
            );
            response = await keywordSearch(sessionId, query, 10);
            actualMethod = "keyword";
            console.log("✅ Keyword fallback successful");
          } catch (fallbackError) {
            console.error("❌ Keyword fallback also failed:", fallbackError);
            throw fallbackError;
          }
        }
      }

      console.log(`✅ ${actualMethod} search response:`, response);

      // Transform response to SearchResult format
      const searchResults =
        response.results
          ?.map((result: any, index: number) => ({
            id: `${result.document}_${result.page_number}_${index}`,
            document: result.document,
            content: result.content,
            page_number: result.page_number,
            score: result.score,
            metadata: {
              ...result.metadata,
              searchMethod: actualMethod,
            },
          }))
          // Filter out results with no content or page markers
          .filter(
            (result: any) =>
              result.content &&
              result.content.trim() &&
              !result.content.includes("--- Page") &&
              result.content.trim() !== "",
          ) || [];

      console.log(`✅ ${actualMethod} search completed:`, searchResults);
      setSearchResults(searchResults);
      setActualSearchMethod(actualMethod);

      // Log if fallback was used
      if (actualMethod !== method) {
        console.log(
          `🔄 Search method changed from ${method} to ${actualMethod} due to fallback`,
        );
      }
    } catch (error) {
      console.error(`❌ All search methods failed:`, {
        error,
        message: error.message,
        stack: error.stack,
      });

      setSearchResults([]);
      setActualSearchMethod(method);
    } finally {
      setIsSearching(false);
    }
  };

  // Debounced search effect
  useEffect(() => {
    if (searchQuery.trim() && sessionId && documents.length > 0) {
      // Clear existing timeout
      if (searchTimeoutRef.current) {
        clearTimeout(searchTimeoutRef.current);
      }

      // Set new timeout for search
      searchTimeoutRef.current = setTimeout(() => {
        console.log("Triggering search:", {
          searchQuery,
          sessionId,
          searchMethod,
          documentsCount: documents.length,
        });
        performSearchWithMethod(searchQuery, sessionId, searchMethod);
        setShowSearchDropdown(true);
      }, 300); // 300ms delay
    } else {
      setShowSearchDropdown(false);
      if (!sessionId) console.log("Search skipped: no session ID");
      if (documents.length === 0) console.log("Search skipped: no documents");
    }

    return () => {
      if (searchTimeoutRef.current) {
        clearTimeout(searchTimeoutRef.current);
      }
    };
  }, [
    searchQuery,
    sessionId,
    searchMethod,
    setShowSearchDropdown,
    documents.length,
  ]);

  // Click outside to close dropdown
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (
        searchContainerRef.current &&
        !searchContainerRef.current.contains(event.target as Node)
      ) {
        setShowSearchDropdown(false);
      }
    }

    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [setShowSearchDropdown]);

  const handleSearchResultClick = (result: any) => {
    console.log("Search result clicked:", result);

    // Find the document by name with more robust matching
    const targetDocument = documents.find((doc) => {
      // First try exact match
      if (doc.name === result.document) return true;

      // Try removing .pdf extension for comparison
      const docNameWithoutExt = doc.name.replace(/\.pdf$/i, "");
      const resultNameWithoutExt = result.document.replace(/\.pdf$/i, "");
      if (docNameWithoutExt === resultNameWithoutExt) return true;

      // Try case-insensitive exact match
      if (doc.name.toLowerCase() === result.document.toLowerCase()) return true;
      if (
        docNameWithoutExt.toLowerCase() === resultNameWithoutExt.toLowerCase()
      )
        return true;

      // Try partial matches (both directions)
      if (
        doc.name.includes(result.document) ||
        result.document.includes(doc.name)
      )
        return true;
      if (
        docNameWithoutExt.includes(resultNameWithoutExt) ||
        resultNameWithoutExt.includes(docNameWithoutExt)
      )
        return true;

      return false;
    });

    if (targetDocument) {
      console.log(
        `WorkspaceHeader: Found target document: ${targetDocument.name}`,
      );

      // Check if we need to switch documents
      const isDocumentSwitch = currentDocument?.name !== targetDocument.name;
      
      // Switch document if needed
      if (isDocumentSwitch) {
        setCurrentDocument(targetDocument);
      }

      // Navigate to the specific page with proper conversion
      // result.page_number is 0-indexed, Adobe expects 1-based
      const adobePageNumber = result.page_number + 1;

      console.log(
        `WorkspaceHeader: Converting page ${result.page_number} (0-indexed) to ${adobePageNumber} (1-based for Adobe)`,
      );

      // Wait for document to load if switching, then navigate with retry logic
      const delay = isDocumentSwitch ? 1500 : 200;

      const dispatchNavigation = (attemptCount = 0) => {
        const maxAttempts = 5;

        try {
          window.dispatchEvent(
            new CustomEvent("go-to-page", {
              detail: { page: adobePageNumber }, // Convert to 1-based
              bubbles: true,
            }),
          );
          console.log(
            `WorkspaceHeader: Dispatched go-to-page event with page ${adobePageNumber} - attempt ${attemptCount + 1}`,
          );
        } catch (error) {
          console.error(
            "WorkspaceHeader: Error dispatching navigation event:",
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
        `WorkspaceHeader: Scheduled navigation to page ${adobePageNumber} (1-based) with ${delay}ms delay${isDocumentSwitch ? " (document switch)" : ""}`,
      );
    } else {
      console.warn(`WorkspaceHeader: Document not found: ${result.document}`);
      console.log(
        "Available documents:",
        documents.map((d) => d.name),
      );
    }

    setShowSearchDropdown(false);
  };
  const exportSession = () => {
    // Placeholder for export functionality
    console.log("Exporting session...");
  };

  const saveSession = () => {
    // Placeholder for save functionality
    console.log("Saving session...");
  };

  return (
    <header className="glass-panel border-b border-border px-6 py-4 relative z-20">
      <div className="flex items-center justify-between">
        {/* Left Section - Branding & Session */}
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <Brain className="h-6 w-6 text-primary glow-red" />
            <h1 className="text-xl font-display font-bold bg-gradient-to-r from-white to-red-300 bg-clip-text text-transparent">
              Connect the Dots
            </h1>
          </div>

          {sessionName && (
            <div className="flex items-center gap-2">
              <span className="text-muted-foreground">•</span>
              <span className="font-medium">{sessionName}</span>
              {isDirty && (
                <Badge variant="secondary" className="text-xs">
                  Unsaved
                </Badge>
              )}
            </div>
          )}

          <Badge variant="outline" className="text-xs">
            {documents.length} document{documents.length !== 1 ? "s" : ""}
          </Badge>
        </div>

        {/* Center Section - Search */}
        <div className="flex-1 max-w-2xl mx-8">
          <div className="relative" ref={searchContainerRef}>
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground" />
            {isSearching && (
              <Loader2 className="absolute right-24 top-1/2 transform -translate-y-1/2 h-4 w-4 text-muted-foreground animate-spin" />
            )}
            <Input
              placeholder={
                !sessionId
                  ? "Create a session first to enable search..."
                  : documents.length === 0
                    ? "Upload documents to enable search..."
                    : `Search across all documents using ${searchMethod} search...`
              }
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onFocus={() => setSearchFocused(true)}
              onBlur={() => setTimeout(() => setSearchFocused(false), 200)}
              disabled={!sessionId || documents.length === 0}
              className={cn(
                "pl-10 pr-24 bg-workspace border-border transition-all duration-200",
                searchFocused && "ring-2 ring-primary/20 border-primary/50",
                (!sessionId || documents.length === 0) &&
                  "opacity-50 cursor-not-allowed",
              )}
            />

            {/* Search Method Toggle */}
            <div className="absolute right-2 top-1/2 transform -translate-y-1/2">
              <Button
                variant="ghost"
                size="sm"
                onClick={() =>
                  setSearchMethod(
                    searchMethod === "semantic" ? "keyword" : "semantic",
                  )
                }
                disabled={!sessionId || documents.length === 0}
                className={cn(
                  "text-xs h-7 px-2 border",
                  searchMethod === "semantic"
                    ? "bg-black border-gray-600 text-white hover:bg-gray-900"
                    : "bg-slate-700/50 border-slate-600 text-slate-300 hover:bg-slate-700/70",
                  (!sessionId || documents.length === 0) &&
                    "opacity-50 cursor-not-allowed",
                )}
              >
                {searchMethod === "semantic" ? "🧠 Semantic" : "🔤 Keyword"}
              </Button>
            </div>

            {/* Search Results Dropdown */}
            {showSearchDropdown && searchResults.length > 0 && (
              <div className="absolute top-full left-0 right-0 mt-1 bg-slate-800 border border-border rounded-lg shadow-xl z-50 max-h-[28rem] overflow-y-auto">
                <div className="p-2">
                  <div className="text-xs text-muted-foreground mb-2 px-2 flex items-center justify-between">
                    <span>Found {searchResults.length} results</span>
                    <div className="flex items-center gap-2">
                      <Badge variant="outline" className="text-xs">
                        {actualSearchMethod === "semantic" && "🧠 Semantic"}
                        {actualSearchMethod === "keyword" && "🔤 Keyword"}
                      </Badge>
                      {actualSearchMethod !== searchMethod && (
                        <Badge
                          variant="secondary"
                          className="text-xs bg-orange-900/50 text-orange-300"
                        >
                          Fallback
                        </Badge>
                      )}
                    </div>
                  </div>
                  {searchResults
                    .filter(result => {
                      // Filter out results with invalid page numbers
                      if (result.page_number === undefined || result.page_number < 0) return false;
                      
                      // Find the document this result belongs to
                      const targetDoc = documents.find(doc => 
                        doc.name === result.document ||
                        doc.name.includes(result.document) ||
                        result.document.includes(doc.name)
                      );
                      
                      // If we can't find the document, keep the result but log warning
                      if (!targetDoc) {
                        console.warn(`Cannot validate page ${result.page_number} - document ${result.document} not found`);
                        return true;
                      }
                      
                      // For now, allow all valid page numbers since we don't have reliable page count
                      // TODO: Add page count validation when available
                      return true;
                    })
                    .map((result) => (
                    <div
                      key={result.id}
                      onClick={() => handleSearchResultClick(result)}
                      className="flex items-start gap-3 p-3 rounded-lg hover:bg-slate-700 cursor-pointer transition-colors border-b border-slate-700/50 last:border-b-0"
                    >
                      <FileText className="h-4 w-4 text-primary mt-1 flex-shrink-0" />
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-sm font-medium text-white truncate">
                            {result.document}
                          </span>
                          <div className="flex items-center gap-2 flex-shrink-0">
                            <Badge variant="outline" className="text-xs">
                              Page {result.page_number + 1}
                            </Badge>
                            <Badge variant="secondary" className="text-xs">
                              {(result.score * 100).toFixed(0)}%
                            </Badge>
                          </div>
                        </div>
                        <div className="space-y-2">
                          <p className="text-xs text-slate-300 leading-relaxed line-clamp-3">
                            {result.content}
                          </p>
                          {result.metadata && (
                            <div className="flex flex-wrap gap-1">
                              {result.metadata.chapter && (
                                <Badge
                                  variant="outline"
                                  className="text-xs bg-slate-700"
                                >
                                  {result.metadata.chapter}
                                </Badge>
                              )}
                              {result.metadata.section && (
                                <Badge
                                  variant="outline"
                                  className="text-xs bg-slate-700"
                                >
                                  {result.metadata.section}
                                </Badge>
                              )}
                              {result.metadata.keywords && (
                                <Badge
                                  variant="outline"
                                  className="text-xs bg-blue-900/50 text-blue-300"
                                >
                                  Keywords:{" "}
                                  {Array.isArray(result.metadata.keywords)
                                    ? result.metadata.keywords
                                        .slice(0, 2)
                                        .join(", ")
                                    : result.metadata.keywords}
                                </Badge>
                              )}
                            </div>
                          )}
                        </div>
                      </div>
                      <ChevronRight className="h-4 w-4 text-muted-foreground flex-shrink-0 mt-1" />
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* No Results State */}
            {showSearchDropdown &&
              searchResults.length === 0 &&
              !isSearching &&
              searchQuery.trim() && (
                <div className="absolute top-full left-0 right-0 mt-1 bg-slate-800 border border-border rounded-lg shadow-xl z-50">
                  <div className="p-4 text-center text-muted-foreground">
                    <Search className="h-8 w-8 mx-auto mb-2 opacity-50" />
                    <p className="text-sm">
                      No results found for "{searchQuery}"
                    </p>
                    {!sessionId && (
                      <p className="text-xs mt-1">
                        Create a session and upload documents to enable search
                      </p>
                    )}
                    {sessionId && documents.length === 0 && (
                      <p className="text-xs mt-1">
                        Upload documents to this session to enable search
                      </p>
                    )}
                  </div>
                </div>
              )}
          </div>
        </div>

        {/* Right Section - Actions & Views */}
        <div className="flex items-center gap-2">
          {/* View Toggle */}
          <div className="flex items-center bg-workspace rounded-lg p-1 border border-border">
            <Button
              variant={currentView === "workspace" ? "default" : "ghost"}
              size="sm"
              onClick={() => setCurrentView("workspace")}
              className="text-xs"
            >
              Workspace
            </Button>
            <Button
              variant={currentView === "knowledge-graph" ? "default" : "ghost"}
              size="sm"
              onClick={() => setCurrentView("knowledge-graph")}
              className="text-xs"
            >
              <Network className="h-3 w-3 mr-1" />
              Graph
            </Button>
          </div>

          {/* Action Buttons */}
          <Button variant="ghost" size="sm" onClick={saveSession}>
            <Save className="h-4 w-4" />
          </Button>

          <Button variant="ghost" size="sm" onClick={exportSession}>
            <Download className="h-4 w-4" />
          </Button>

          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" size="sm">
                <MoreHorizontal className="h-4 w-4" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="bg-panel border-border">
              <DropdownMenuItem onClick={() => setCurrentView("upload")}>
                <Upload className="h-4 w-4 mr-2" />
                Add Documents
              </DropdownMenuItem>
              <DropdownMenuItem>
                <Settings className="h-4 w-4 mr-2" />
                Settings
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem onClick={exportSession}>
                <Download className="h-4 w-4 mr-2" />
                Export Session
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </div>
    </header>
  );
}
