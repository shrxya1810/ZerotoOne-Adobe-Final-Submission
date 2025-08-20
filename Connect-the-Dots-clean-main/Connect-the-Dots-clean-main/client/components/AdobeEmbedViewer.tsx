import React, { useImperativeHandle, forwardRef } from "react";
import { useEffect, useRef } from "react";
// Add AdobeDC type to window for TypeScript
declare global {
  interface Window {
    AdobeDC?: any;
    adobe_dc_view_sdk_ready?: boolean;
  }
}

// Adobe Embed API expects the script to be loaded in index.html
// and a div with a specific id to mount the viewer.
// Client ID should be provided via VITE_ADOBE_CLIENT_ID env variable.

export interface AdobeEmbedViewerHandle {
  gotoPage: (page: number) => void;
  gotoPageAndHighlight: (page: number, searchText?: string) => void;
  searchAndHighlight: (searchText: string) => void;
}

interface AdobeEmbedViewerProps {
  fileName: string; // The PDF filename or URL to fetch from your backend
  docId: string; // Document ID for insights
  sessionId: string; // Session ID for insights
  onSelection?: (text: string, pageNum?: number) => void;
}

export const AdobeEmbedViewer = forwardRef<
  AdobeEmbedViewerHandle,
  AdobeEmbedViewerProps
>(({ fileName, docId, sessionId, onSelection }, ref) => {
  const viewerRef = useRef<HTMLDivElement>(null);
  const adobeViewRef = useRef<any>(null);
  const adobeApisRef = useRef<any>(null);
  const selectionWatcherRef = useRef<any>(null);
  const lastSelectionRef = useRef<string>("");

  // Ensure the viewer div has a unique id and use it for AdobeDC.View
  useEffect(() => {
    if (viewerRef.current && !viewerRef.current.id) {
      viewerRef.current.id = `adobe-dc-view-${Math.random().toString(36).slice(2)}`;
    }
  }, []);

  useEffect(() => {
    let cancelled = false;
    function onReady() {
      if (!window.AdobeDC || !viewerRef.current) return;
      const clientId = import.meta.env.VITE_ADOBE_CLIENT_ID;
      if (!clientId) {
        console.error("Adobe Client ID missing in env");
        return;
      }
      const view = new window.AdobeDC.View({
        clientId,
        divId: viewerRef.current.id,
      });
      adobeViewRef.current = view;
      const filePromise = fetch(
        `/api/pdfs/${sessionId}/${encodeURIComponent(fileName)}`,
      ).then((r) => {
        if (!r.ok) throw new Error("PDF fetch failed " + r.status);
        return r.arrayBuffer();
      });
      view
        .previewFile(
          {
            content: { promise: filePromise },
            metaData: { fileName },
          },
          { 
            embedMode: "SIZED_CONTAINER",
            defaultViewMode: "FIT_PAGE",
            showDownloadPDF: false,
            showPrintPDF: false,
            showLeftHandPanel: false,
            showAnnotationTools: false,
            enableSearchAPIs: true
          },
        )
        .then((instance: any) => {
          if (cancelled) return;
          return instance.getAPIs();
        })
        .then((apis: any) => {
          if (cancelled) return;
          adobeApisRef.current = apis;
          startSelectionWatcher();
        })
        .catch((err: any) => {
          if (!cancelled) console.error("Adobe preview failed:", err);
        });
    }

    if (window.AdobeDC && window.AdobeDC.View) {
      onReady();
    } else {
      const handler = () => {
        onReady();
      };
      document.addEventListener("adobe_dc_view_sdk.ready", handler, {
        once: true,
      });
      return () => {
        document.removeEventListener("adobe_dc_view_sdk.ready", handler);
        cancelled = true;
        stopSelectionWatcher();
      };
    }
    return () => {
      cancelled = true;
      stopSelectionWatcher();
    };
    // eslint-disable-next-line
  }, [fileName]);

  function startSelectionWatcher() {
    stopSelectionWatcher();
    selectionWatcherRef.current = setInterval(async () => {
      const apis = adobeApisRef.current;
      if (!apis) return;
      try {
        const res = await apis.getSelectedContent();
        let txt = "";
        let pg = undefined;
        if (res?.data && typeof res.data === "string") {
          txt = res.data.trim();
        } else if (Array.isArray(res?.data)) {
          txt = res.data
            .map((p) => p?.text || "")
            .join("\n")
            .trim();
          const any = res.data.find((p) => p?.path?.pageNumber);
          pg = any ? any.path.pageNumber : undefined;
        } else if (res?.text) {
          txt = res.text.trim();
        }
        // Only trigger if selection actually changed
        if (txt && txt !== lastSelectionRef.current && onSelection) {
          lastSelectionRef.current = txt;
          onSelection(txt, pg);
        } else if (!txt && lastSelectionRef.current) {
          // Selection was cleared
          lastSelectionRef.current = "";
        }
      } catch {}
    }, 400);
  }
  function stopSelectionWatcher() {
    if (selectionWatcherRef.current) {
      clearInterval(selectionWatcherRef.current);
      selectionWatcherRef.current = null;
    }
  }

  // Zoom and go-to-page controls
  const zoomIn = () => {
    adobeApisRef.current?.zoomIn?.().catch(() => {});
  };
  const zoomOut = () => {
    adobeApisRef.current?.zoomOut?.().catch(() => {});
  };

  // Expose navigation and highlighting methods to parent via ref
  useImperativeHandle(ref, () => ({
    gotoPage: (page: number) => {
      console.log(`AdobeEmbedViewer: gotoPage called with page ${page}`);

      // Basic validation - page must be a positive number
      if (!page || page < 1 || page > 500) {
        // Reasonable upper limit
        console.warn(
          `Invalid page number: ${page}. Must be between 1 and 500.`,
        );
        return;
      }

      if (adobeApisRef.current?.gotoLocation) {
        adobeApisRef.current
          .gotoLocation(page)
          .then(() => {
            console.log(`Successfully navigated to page ${page}`);
          })
          .catch((error) => {
            console.error(`Failed to navigate to page ${page}:`, error);
            // If navigation fails, it might be because page doesn't exist
            if (error.message?.includes("INVALID_INPUT")) {
              console.warn(`Page ${page} does not exist in this document`);
            }
          });
      } else {
        console.warn("Adobe APIs not available for navigation");
      }
    },

    gotoPageAndHighlight: (page: number, searchText?: string) => {
      console.log(`AdobeEmbedViewer: gotoPageAndHighlight called with page ${page} and text "${searchText}"`);
      
      // First navigate to the page
      if (adobeApisRef.current?.gotoLocation) {
        adobeApisRef.current
          .gotoLocation(page)
          .then(() => {
            console.log(`Successfully navigated to page ${page}`);
            
            // Then search and highlight the text if provided
            if (searchText && adobeApisRef.current?.search) {
              setTimeout(() => {
                adobeApisRef.current.search(searchText)
                  .then((searchResult: any) => {
                    console.log(`Search completed for "${searchText}":`, searchResult);
                  })
                  .catch((error: any) => {
                    console.warn(`Search failed for "${searchText}":`, error);
                  });
              }, 1000); // Wait for page navigation to complete
            }
          })
          .catch((error) => {
            console.error(`Failed to navigate to page ${page}:`, error);
          });
      }
    },

    searchAndHighlight: (searchText: string) => {
      console.log(`AdobeEmbedViewer: searchAndHighlight called with text "${searchText}"`);
      
      if (adobeApisRef.current?.search) {
        adobeApisRef.current.search(searchText)
          .then((searchResult: any) => {
            console.log(`Search completed for "${searchText}":`, searchResult);
          })
          .catch((error: any) => {
            console.warn(`Search failed for "${searchText}":`, error);
          });
      } else {
        console.warn("Adobe search API not available");
      }
    },
  }));

  return (
    <div>
      <div
        ref={viewerRef}
        style={{ height: 720, border: "1px solid #e5e7eb", borderRadius: 12 }}
      ></div>
    </div>
  );
});

// Make sure to add <script src="https://documentcloud.adobe.com/view-sdk/main.js"></script> in your index.html
// and set VITE_ADOBE_CLIENT_ID in your .env file.
