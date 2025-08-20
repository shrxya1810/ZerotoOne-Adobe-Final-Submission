import JourneyBar from '../components/JourneyBar';
import { useState, useEffect, useRef, useCallback } from 'react';
import { useAppStore } from '../lib/store';
import { Button } from '../components/ui/button';
import { Input } from '../components/ui/input';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/card';

import { Separator } from '../components/ui/separator';
import { Network, ArrowLeft, Search, Settings, ZoomIn, ZoomOut, Download, Loader2 } from 'lucide-react';
import ForceGraph2D from 'react-force-graph-2d';
import { generateKnowledgeGraph, searchGraphNodes, getGraphStatistics, getRelationshipDetails, GraphNode, GraphEdge, KnowledgeGraphResponse } from '@shared/api';

// Walkthrough steps definition
const WALKTHROUGH_STEPS = [
  {
    title: 'Knowledge Graph',
    description: 'This interactive graph shows connections between concepts, topics, and entities across your research documents.',
    target: 'graph',
  },
  {
    title: 'Back to Workspace',
    description: 'Use this button to return to your main research workspace at any time.',
    target: 'back',
  },
];

// Node colors by type
const NODE_COLORS = {
  document: '#3b82f6',    // Blue for documents
};

// Community colors
const COMMUNITY_COLORS = [
  '#3b82f6', '#10b981', '#8b5cf6', '#f59e0b', '#ef4444',
  '#06b6d4', '#84cc16', '#f97316', '#ec4899', '#6366f1'
];

// Centrality visualization
const getCentralityColor = (centrality: number) => {
  const intensity = Math.min(centrality * 3, 1); // Scale for visibility
  return `rgba(255, 215, 0, ${0.3 + intensity * 0.7})`; // Gold with varying opacity
};

const getCentralitySize = (centrality: number, baseSize: number) => {
  return baseSize + (centrality * 20); // Scale size based on centrality
};

// Node sizes by type
const NODE_SIZES = {
  document: 10
};

// Edge colors by relationship type - Enhanced for better visibility
const EDGE_COLORS = {
  contains: '#60a5fa',        // Bright blue for session->document
  highly_similar: '#10b981',  // Green for high similarity
  similar_to: '#3b82f6',      // Blue for medium similarity
  related_to: '#a855f7',      // Brighter purple for better visibility
  share_concept: '#f59e0b',   // Orange for shared concepts
  share_entity: '#ef4444',    // Red for shared entities
  mentions: '#60a5fa',        // Bright blue for mentions
  discusses: '#60a5fa'        // Bright blue for discusses
};

interface ForceGraphData {
  nodes: Array<GraphNode & { x?: number; y?: number; vx?: number; vy?: number; fx?: number; fy?: number; color?: string; size?: number }>;
  links: Array<GraphEdge & { source: string; target: string; color?: string }>;
}

export default function KnowledgeGraphPage() {
  const { setCurrentView, sessionId, setSelectedGraphNode, setCurrentStep } = useAppStore();

  // Highlight Knowledge Graph step in JourneyBar when this page is loaded
  useEffect(() => {
    setCurrentStep(4); // 5th icon (0-based)
    return () => {
      setCurrentStep(3); // Go back to previous step (workspace) when leaving
    };
  }, [setCurrentStep]);
  const fgRef = useRef<any>();
  
  // Component state
  const [walkthroughStep, setWalkthroughStep] = useState<number | null>(() => {
    const hasSeenWalkthrough = localStorage.getItem('hasSeenGraphWalkthrough');
    return hasSeenWalkthrough ? null : 0;
  });
  const [graphData, setGraphData] = useState<ForceGraphData>({ nodes: [], links: [] });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);

  const [graphStats, setGraphStats] = useState<any>(null);
  const [selectedEdge, setSelectedEdge] = useState<GraphEdge | null>(null);
  const [edgeDetails, setEdgeDetails] = useState<any>(null);
  const [showRelationshipModal, setShowRelationshipModal] = useState(false);


  const [analyticsData, setAnalyticsData] = useState<any>(null);
  const [visualizationMode, setVisualizationMode] = useState<'default' | 'centrality' | 'community'>('default');

  // Load knowledge graph data
  const loadGraph = useCallback(async () => {
    if (!sessionId) return;
    
    setLoading(true);
    setError(null);
    
    try {
      // Always use a low threshold to get ALL edges from backend
      // The backend will create edges based on its own logic, frontend shows everything
      const response = await generateKnowledgeGraph(
        sessionId, 
        0.3 // Fixed low threshold to ensure all edges are created
      );
      
      // Store analytics data
      try {
        setAnalyticsData({
          centrality: response.graph_stats?.centrality_metrics || {},
          communities: response.graph_stats?.communities || {},
          communitiesCount: response.graph_stats?.communities_count || 0
        });
      } catch (error) {
        console.error('Error setting analytics data:', error);
        setAnalyticsData({
          centrality: {},
          communities: {},
          communitiesCount: 0
        });
      }
      
      // Transform data for ForceGraph with static positioning
      const documentNodes = response.nodes.filter(node => node.type === 'document');
      
      const transformedData: ForceGraphData = {
        nodes: response.nodes.map((node, index) => {
          const centrality = response.graph_stats?.centrality_metrics?.[node.id]?.composite || 0;
          const community = response.graph_stats?.communities?.[node.id] || 0;
          
          let color = NODE_COLORS[node.type] || '#6b7280';
          let size = NODE_SIZES[node.type] || 6;
          
          // Apply visualization mode
          if (visualizationMode === 'centrality' && node.type === 'document') {
            color = getCentralityColor(centrality);
            size = getCentralitySize(centrality, NODE_SIZES[node.type] || 6);
          } else if (visualizationMode === 'community' && node.type === 'document') {
            color = COMMUNITY_COLORS[community % COMMUNITY_COLORS.length];
          }
          
          // Static positioning to prevent overlap
          let x = 0, y = 0;
          if (node.type === 'document') {
            const docIndex = documentNodes.findIndex(doc => doc.id === node.id);
            const angle = (docIndex / documentNodes.length) * 2 * Math.PI;
            const radius = documentNodes.length > 2 ? 150 : 80;
            x = Math.cos(angle) * radius;
            y = Math.sin(angle) * radius;
          }
          
          return {
            ...node,
            color,
            size,
            centrality,
            community,
            x,
            y,
            fx: x, // Fixed x position
            fy: y  // Fixed y position
          };
        }),
        links: response.edges.map(edge => ({
          ...edge,
          color: EDGE_COLORS[edge.relationship] || '#60a5fa'
        }))
      };
      
      setGraphData(transformedData);
      
      // Load graph statistics
      const stats = await getGraphStatistics(sessionId);
      setGraphStats(stats);
      
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load knowledge graph');
    } finally {
      setLoading(false);
    }
  }, [sessionId]);

  // Load graph on component mount and when settings change
  useEffect(() => {
    loadGraph();
  }, [loadGraph]);
  


  // Update visualization when mode changes
  useEffect(() => {
    if (graphData.nodes.length > 0 && analyticsData) {
      try {
        const updatedNodes = graphData.nodes.map(node => {
          const centrality = analyticsData.centrality?.[node.id]?.composite || 0;
          const community = analyticsData.communities?.[node.id] || 0;
          
          let color = NODE_COLORS[node.type] || '#6b7280';
          let size = NODE_SIZES[node.type] || 6;
          
          if (visualizationMode === 'centrality' && node.type === 'document') {
            color = getCentralityColor(centrality);
            size = getCentralitySize(centrality, NODE_SIZES[node.type] || 6);
          } else if (visualizationMode === 'community' && node.type === 'document') {
            color = COMMUNITY_COLORS[community % COMMUNITY_COLORS.length];
          }
          
          // Maintain fixed positions
          return { 
            ...node, 
            color, 
            size,
            fx: node.fx || node.x,
            fy: node.fy || node.y
          };
        });
        
        setGraphData(prev => ({ ...prev, nodes: updatedNodes }));
      } catch (error) {
        console.error('Error updating visualization:', error);
      }
    }
  }, [visualizationMode, analyticsData]);

  // Handle node click
  const handleNodeClick = useCallback(async (node: any) => {
    if (!sessionId) return;
    
    setSelectedNode(node);
    setSelectedGraphNode(node);
  }, [sessionId, setSelectedGraphNode]);

  // Handle edge click
  const handleEdgeClick = useCallback(async (edge: any) => {
    if (!sessionId) return;
    
    setSelectedEdge(edge);
    setShowRelationshipModal(true);
    setEdgeDetails(null); // Clear previous details
    
    try {
      // Extract the actual node IDs - edge might be from ForceGraph with source/target objects
      const sourceId = typeof edge.source === 'object' ? edge.source.id : edge.source;
      const targetId = typeof edge.target === 'object' ? edge.target.id : edge.target;
      
      console.log('🔗 Fetching relationship details for:', sourceId, '→', targetId);
      const details = await getRelationshipDetails(sessionId, sourceId, targetId);
      console.log('🔗 Received relationship details:', details);
      setEdgeDetails(details);
    } catch (err) {
      console.error('Failed to load edge details:', err);
      // Set fallback details based on the edge data
      setEdgeDetails({
        relationship: { ...edge },
        source_node: { label: 'Unknown', id: edge.source },
        target_node: { label: 'Unknown', id: edge.target },
        summary: 'Failed to load relationship details',
        strength: edge.weight || 0,
        type: edge.relationship
      });
    }
  }, [sessionId]);

  // Handle search
  const handleSearch = async () => {
    if (!sessionId || !searchQuery.trim()) return;
    
    try {
      const results = await searchGraphNodes(sessionId, searchQuery);
      if (results.matching_nodes.length > 0) {
        const firstMatch = results.matching_nodes[0];
        // Find and focus on the matching node
        const node = graphData.nodes.find(n => n.id === firstMatch.id);
        if (node && fgRef.current) {
          fgRef.current.centerAt(node.x, node.y, 1000);
          fgRef.current.zoom(2, 2000);
          setSelectedNode(node);
        }
      }
    } catch (err) {
      console.error('Search failed:', err);
    }
  };

  // Walkthrough handlers
  const handleNextWalkthrough = () => {
    if (walkthroughStep !== null && walkthroughStep < WALKTHROUGH_STEPS.length - 1) {
      setWalkthroughStep(walkthroughStep + 1);
    } else {
      setWalkthroughStep(null);
      localStorage.setItem('hasSeenGraphWalkthrough', 'true');
    }
  };
  const handleSkipWalkthrough = () => {
    setWalkthroughStep(null);
    localStorage.setItem('hasSeenGraphWalkthrough', 'true');
  };

  // Graph controls
  const handleZoomIn = () => fgRef.current?.zoom(fgRef.current.zoom() * 1.5, 400);
  const handleZoomOut = () => fgRef.current?.zoom(fgRef.current.zoom() * 0.75, 400);
  const handleFitToView = () => fgRef.current?.zoomToFit(400);

  return (
    <div className="min-h-screen bg-workspace text-workspace-foreground">
      {/* Journey Bar */}
      <JourneyBar currentStep={4} />


            






                

                

      {/* Relationship Details Modal */}
      {showRelationshipModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
          <div className="bg-slate-800 rounded-2xl shadow-2xl p-6 max-w-2xl w-full max-h-[80vh] overflow-y-auto border border-blue-400 animate-fade-in">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-xl font-bold text-white flex items-center gap-2">
                <div 
                  className="w-4 h-1 rounded" 
                  style={{ backgroundColor: selectedEdge ? EDGE_COLORS[selectedEdge.relationship] || '#6b7280' : '#6b7280' }}
                />
                Relationship Details
              </h3>
              <Button 
                variant="ghost" 
                size="sm" 
                onClick={() => setShowRelationshipModal(false)}
                className="text-slate-400 hover:text-white"
              >
                ✕
              </Button>
            </div>
            
            {edgeDetails ? (
              <div className="space-y-4">
                <div className="flex items-center gap-4 p-3 bg-slate-700/50 rounded-lg">
                  <div className="text-center">
                    <div className="text-sm text-slate-400">From</div>
                    <div className="font-medium text-blue-400">
                      {edgeDetails.source_node?.label || 'Unknown'}
                    </div>
                  </div>
                  <div className="flex-1 text-center">
                    <div 
                      className="h-1 rounded mx-4" 
                      style={{ backgroundColor: selectedEdge ? EDGE_COLORS[selectedEdge.relationship] || '#6b7280' : '#6b7280' }}
                    />
                    <div className="text-sm text-slate-400 mt-1">
                      {edgeDetails.type?.replace('_', ' ') || 'Related'}
                    </div>
                  </div>
                  <div className="text-center">
                    <div className="text-sm text-slate-400">To</div>
                    <div className="font-medium text-blue-400">
                      {edgeDetails.target_node?.label || 'Unknown'}
                    </div>
                  </div>
                </div>
                
                <div className="p-4 bg-slate-700/30 rounded-lg">
                  <h4 className="font-medium mb-2 text-slate-200">Relationship Summary</h4>
                  <p className="text-slate-300 leading-relaxed">
                    {edgeDetails.summary || 'No summary available.'}
                  </p>
                </div>
                
                <div className="flex justify-between items-center text-sm text-slate-400">
                  <span>Strength: {((edgeDetails.strength || 0) * 100).toFixed(0)}%</span>
                  <span>Type: {edgeDetails.type?.replace('_', ' ') || 'Unknown'}</span>
                </div>
              </div>
            ) : (
              <div className="text-center py-8">
                <Loader2 className="h-8 w-8 animate-spin mx-auto mb-4 text-blue-400" />
                <p className="text-slate-400">Loading relationship details...</p>
              </div>
            )}
          </div>
        </div>
      )}

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
              <Button onClick={handleSkipWalkthrough} variant="ghost" className="text-slate-400 hover:text-red-400">Skip</Button>
              <Button onClick={handleNextWalkthrough} className="glass-button text-white font-semibold px-6 py-2 rounded-lg">
                {walkthroughStep === WALKTHROUGH_STEPS.length - 1 ? 'Finish' : 'Next'}
              </Button>
            </div>
            <div className="absolute top-2 right-4 text-xs text-slate-400">{walkthroughStep + 1}/{WALKTHROUGH_STEPS.length}</div>
          </div>
        </div>
      )}

      {/* Header */}
      <div className="border-b border-border/40 bg-card/30 backdrop-blur-sm">
        <div className="flex items-center justify-between p-6 overflow-hidden">
          <div className="flex items-center gap-4">
            <Button 
              variant="ghost" 
              onClick={() => setCurrentView('workspace')}
              className={walkthroughStep !== null && WALKTHROUGH_STEPS[walkthroughStep].target === 'back' ? 
                'flex items-center gap-2 ring-4 ring-red-400 ring-offset-2 ring-offset-workspace' : 
                'flex items-center gap-2'
              }
            >
              <ArrowLeft className="h-4 w-4" />
              Back to Workspace
            </Button>
            <div className="min-w-0">
              <h1 className="text-2xl font-bold flex items-center gap-2">
                <Network className="h-6 w-6 text-red-400 flex-shrink-0" />
                <span className="truncate">Knowledge Graph</span>
              </h1>
              {graphStats && (
                <p className="text-sm text-muted-foreground mt-1 truncate">
                  {graphStats.total_stats.total_nodes} nodes • {graphStats.total_stats.total_edges} connections
                </p>
              )}
            </div>
          </div>
          
          {/* Search and Controls */}
          <div className="flex items-center gap-3 overflow-hidden flex-shrink-0">
            <div className="flex items-center gap-2 min-w-0">
              <Input
                placeholder="Search..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
                className="w-40 min-w-0 text-sm"
              />
              <Button size="sm" variant="outline" onClick={handleSearch}>
                <Search className="h-3 w-3" />
              </Button>
            </div>
            
            <Separator orientation="vertical" className="h-6" />
            
            <div className="flex items-center gap-1">
              <Button size="sm" variant="outline" onClick={handleZoomIn}>
                <ZoomIn className="h-4 w-4" />
              </Button>
              <Button size="sm" variant="outline" onClick={handleZoomOut}>
                <ZoomOut className="h-4 w-4" />
              </Button>
              <Button size="sm" variant="outline" onClick={handleFitToView}>
                Fit
              </Button>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex h-[calc(100vh-88px)] overflow-hidden">
        {/* Graph Container */}
        <div 
          className={walkthroughStep !== null && WALKTHROUGH_STEPS[walkthroughStep].target === 'graph' ? 
            'flex-1 relative ring-4 ring-red-400 ring-offset-2 ring-offset-workspace min-w-0' : 
            'flex-1 relative min-w-0'
          }
        >
          {loading ? (
            <div className="absolute inset-0 flex items-center justify-center bg-card/20 backdrop-blur-sm">
              <div className="text-center">
                <Loader2 className="h-8 w-8 animate-spin mx-auto mb-4 text-red-400" />
                <p className="text-muted-foreground">Generating knowledge graph...</p>
              </div>
            </div>
          ) : error ? (
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="text-center">
                <Network className="h-16 w-16 mx-auto mb-4 text-muted-foreground opacity-30" />
                <h3 className="text-lg font-medium mb-2">Failed to load graph</h3>
                <p className="text-red-400 mb-4">{error}</p>
                <Button onClick={loadGraph} variant="outline">
                  Retry
                </Button>
              </div>
            </div>
          ) : graphData.nodes.length === 0 ? (
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="text-center">
                <Network className="h-16 w-16 mx-auto mb-4 text-muted-foreground opacity-30" />
                <h3 className="text-lg font-medium mb-2">No Data Available</h3>
                <p className="text-muted-foreground max-w-md">
                  Upload some documents first to generate a knowledge graph showing connections between concepts and entities.
                </p>
              </div>
            </div>
          ) : (
            <div className="w-full h-full">
              <ForceGraph2D
                ref={fgRef}
                graphData={graphData}
                width={undefined}
                height={undefined}
                nodeLabel={(node: any) => `${node.label} (${node.type})`}
                nodeColor={(node: any) => node.color}
                nodeVal={(node: any) => node.size}
                linkColor={(link: any) => {
                  // Ensure edge colors are always visible with better contrast
                  const color = link.color || '#60a5fa';
                  
                  // Add fallback for low-contrast colors and ensure visibility
                  if (!color || color === '#6b7280') {
                    return '#60a5fa'; // Bright blue fallback
                  }
                  
                  // Ensure minimum contrast by checking if color is too dark
                  if (color === '#8b5cf6' && link.weight < 0.6) {
                    return '#a855f7'; // Brighter purple for low-weight edges
                  }
                  
                  return color;
                }}
                linkWidth={(link: any) => {
                  // Make edges visible based on threshold, not weight
                  const relationship = link.relationship;
                  
                  // Set minimum widths based on relationship type for better visibility
                  if (relationship === 'highly_similar') {
                    return 4; // Thick for high similarity
                  } else if (relationship === 'similar_to') {
                    return 3; // Medium for similar
                  } else if (relationship === 'related_to') {
                    return 2; // Visible for related
                  } else {
                    return 2; // Default minimum width
                  }
                }}
                linkLabel={(link: any) => `${link.relationship} (${(link.weight * 100).toFixed(0)}%)`}
                onNodeClick={handleNodeClick}
                onLinkClick={handleEdgeClick}
                nodeCanvasObject={(node: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
                  const label = node.label;
                  const fontSize = 12/globalScale;
                  ctx.font = `${fontSize}px Sans-Serif`;
                  
                  // Draw node with subtle shadow
                  ctx.fillStyle = 'rgba(0,0,0,0.1)';
                  ctx.beginPath();
                  ctx.arc(node.x + 1, node.y + 1, node.size, 0, 2 * Math.PI, false);
                  ctx.fill();
                  
                  // Draw main node
                  ctx.fillStyle = node.color;
                  ctx.beginPath();
                  ctx.arc(node.x, node.y, node.size, 0, 2 * Math.PI, false);
                  ctx.fill();
                  
                  // Draw subtle border
                  ctx.strokeStyle = 'rgba(255,255,255,0.3)';
                  ctx.lineWidth = 1;
                  ctx.stroke();
                  
                  // Draw label with better contrast
                  if (globalScale > 1) {
                    ctx.textAlign = 'center';
                    ctx.textBaseline = 'middle';
                    
                    // Text shadow for better readability
                    ctx.fillStyle = 'rgba(0,0,0,0.8)';
                    ctx.fillText(label, node.x + 1, node.y + node.size + fontSize + 1);
                    
                    ctx.fillStyle = '#ffffff';
                    ctx.fillText(label, node.x, node.y + node.size + fontSize);
                  }
                }}

                cooldownTicks={0}
                warmupTicks={0}
                d3AlphaMin={0}
                d3AlphaDecay={1}
                enableNodeDrag={false}
                enableZoomInteraction={true}
                enablePanInteraction={true}
                backgroundColor="rgba(0,0,0,0)"
              />
            </div>
          )}
        </div>

        {/* Side Panel - Always Visible */}
        <div 
          className="w-80 flex-shrink-0 border-l border-border/40 bg-gradient-to-b from-slate-800/90 to-slate-900/90 backdrop-blur-sm flex flex-col shadow-xl"
          style={{ minWidth: '320px', maxWidth: '320px', height: '100%' }}
        >
          <div className="flex-1 overflow-y-auto overflow-x-hidden">
          {/* Graph Settings */}
          <div className="p-4 border-b border-slate-700/50">
            <h3 className="font-semibold mb-3 text-slate-100 flex items-center gap-2">
              <div className="w-2 h-2 bg-purple-400 rounded-full"></div>
              Graph Settings
            </h3>
            <div className="space-y-3">
              <div className="text-center p-3 bg-slate-700/30 rounded">
                <p className="text-xs text-slate-300 mb-2">
                  Backend creates all edges automatically
                </p>
                <p className="text-xs text-slate-400">
                  Frontend displays everything the backend generates
                </p>
              </div>
              <Button
                size="sm"
                variant="outline"
                onClick={loadGraph}
                className="w-full text-xs"
              >
                Regenerate Graph
              </Button>
            </div>
          </div>

          {/* Analytics Controls */}
          <div className="p-4 border-b border-slate-700/50">
            <h3 className="font-semibold mb-3 text-slate-100 flex items-center gap-2">
              <div className="w-2 h-2 bg-purple-400 rounded-full"></div>
              Analytics
            </h3>
            <div className="space-y-2">
              <button
                onClick={() => setVisualizationMode('default')}
                className={`w-full text-left px-2 py-1 rounded text-xs transition-colors ${
                  visualizationMode === 'default' 
                    ? 'bg-blue-500/30 text-blue-300 border border-blue-400/50' 
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-700/30'
                }`}
              >
                Default View
              </button>
              <button
                onClick={() => setVisualizationMode('centrality')}
                className={`w-full text-left px-2 py-1 rounded text-xs transition-colors ${
                  visualizationMode === 'centrality' 
                    ? 'bg-yellow-500/30 text-yellow-300 border border-yellow-400/50' 
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-700/30'
                }`}
              >
                Document Importance
              </button>
              <button
                onClick={() => setVisualizationMode('community')}
                className={`w-full text-left px-2 py-1 rounded text-xs transition-colors ${
                  visualizationMode === 'community' 
                    ? 'bg-green-500/30 text-green-300 border border-green-400/50' 
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-700/30'
                }`}
              >
                Research Communities
              </button>
            </div>
          </div>
          
          {/* Legend */}
          <div className="p-4 border-b border-slate-700/50">
            <h3 className="font-semibold mb-3 text-slate-100 flex items-center gap-2">
              <div className="w-2 h-2 bg-blue-400 rounded-full"></div>
              Legend
            </h3>
            
            {/* Legend Content Based on Mode */}
            {visualizationMode === 'default' && (
              <div className="mb-4">
                <h4 className="text-sm font-medium mb-2 text-slate-300">Graph Type</h4>
                <div className="p-2 rounded bg-slate-700/30">
                  <div className="flex items-center gap-2">
                    <div 
                      className="w-3 h-3 rounded-full shadow-sm" 
                      style={{ backgroundColor: NODE_COLORS.document }}
                    />
                    <span className="text-xs text-slate-200">Document Relationships</span>
                  </div>
                  <p className="text-xs text-slate-400 mt-1">Connections based on content similarity</p>
                </div>
              </div>
            )}
            
            {visualizationMode === 'centrality' && (
              <div className="mb-4">
                <h4 className="text-sm font-medium mb-2 text-slate-300">Importance Scale</h4>
                <div className="space-y-2">
                  <div className="flex items-center gap-2 p-1.5 rounded bg-slate-700/30">
                    <div className="w-3 h-3 rounded-full bg-yellow-400 opacity-30"></div>
                    <span className="text-xs text-slate-300">Low Importance</span>
                  </div>
                  <div className="flex items-center gap-2 p-1.5 rounded bg-slate-700/30">
                    <div className="w-4 h-4 rounded-full bg-yellow-400 opacity-60"></div>
                    <span className="text-xs text-slate-300">Medium Importance</span>
                  </div>
                  <div className="flex items-center gap-2 p-1.5 rounded bg-slate-700/30">
                    <div className="w-5 h-5 rounded-full bg-yellow-400"></div>
                    <span className="text-xs text-slate-300">High Importance</span>
                  </div>
                </div>
              </div>
            )}
            
            {visualizationMode === 'community' && (
              <div className="mb-4">
                <h4 className="text-sm font-medium mb-2 text-slate-300">Community Colors</h4>
                <div className="grid grid-cols-2 gap-2">
                  {COMMUNITY_COLORS.slice(0, 6).map((color, index) => (
                    <div key={index} className="flex items-center gap-2 p-1.5 rounded bg-slate-700/30">
                      <div 
                        className="w-3 h-3 rounded-full shadow-sm" 
                        style={{ backgroundColor: color }}
                      />
                      <span className="text-xs text-slate-300">Group {index + 1}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Relationship Types */}
            <div>
              <h4 className="text-sm font-medium mb-2 text-slate-300">Relationships</h4>
              <div className="space-y-1.5">
                <div className="flex items-center gap-2 p-1 rounded">
                  <div className="w-4 h-0.5 rounded-full" style={{ backgroundColor: EDGE_COLORS.highly_similar }} />
                  <span className="text-xs text-slate-300">Highly Similar</span>
                </div>
                <div className="flex items-center gap-2 p-1 rounded">
                  <div className="w-4 h-0.5 rounded-full" style={{ backgroundColor: EDGE_COLORS.similar_to }} />
                  <span className="text-xs text-slate-300">Similar</span>
                </div>
                <div className="flex items-center gap-2 p-1 rounded">
                  <div className="w-4 h-0.5 rounded-full" style={{ backgroundColor: EDGE_COLORS.related_to }} />
                  <span className="text-xs text-slate-300">Related</span>
                </div>
                <div className="flex items-center gap-2 p-1 rounded">
                  <div className="w-4 h-0.5 rounded-full" style={{ backgroundColor: EDGE_COLORS.contains }} />
                  <span className="text-xs text-slate-300">Contains</span>
                </div>
                <div className="flex items-center gap-2 p-1 rounded">
                  <div className="w-4 h-0.5 rounded-full" style={{ backgroundColor: '#60a5fa' }} />
                  <span className="text-xs text-slate-300">Other Connections</span>
                </div>
              </div>
            </div>
          </div>

          {/* Analytics Insights */}
          <div className="p-4 border-b border-slate-700/50">
            <h3 className="font-semibold mb-3 text-slate-100 flex items-center gap-2">
              <div className="w-2 h-2 bg-green-400 rounded-full"></div>
              Insights
            </h3>
            
            {loading ? (
              <div className="bg-slate-700/30 p-4 rounded text-center">
                <div className="animate-spin w-6 h-6 border-2 border-green-400 border-t-transparent rounded-full mx-auto mb-2"></div>
                <div className="text-slate-400 text-xs">Analyzing documents...</div>
              </div>
            ) : analyticsData ? (
              <>
                {visualizationMode === 'centrality' && analyticsData.centrality && Object.keys(analyticsData.centrality).length > 0 && (
                  <div className="space-y-2 mb-4">
                    <h4 className="text-xs font-medium text-slate-300 mb-2">Most Important Documents</h4>
                    {Object.entries(analyticsData.centrality)
                      .sort(([,a]: [string, any], [,b]: [string, any]) => b.composite - a.composite)
                      .slice(0, 3)
                      .map(([nodeId, metrics]: [string, any], index) => {
                        const node = graphData.nodes.find(n => n.id === nodeId);
                        return (
                          <div key={nodeId} className="bg-yellow-500/10 border border-yellow-400/30 p-2 rounded text-xs">
                            <div className="text-yellow-300 font-medium truncate">{node?.label || nodeId}</div>
                            <div className="text-yellow-400/70">Importance: {(metrics.composite * 100).toFixed(1)}%</div>
                          </div>
                        );
                      })}
                  </div>
                )}
                
                {visualizationMode === 'community' && (
                  <div className="space-y-2 mb-4">
                    <h4 className="text-xs font-medium text-slate-300 mb-2">Research Communities</h4>
                    <div className="bg-green-500/10 border border-green-400/30 p-2 rounded text-xs">
                      <div className="text-green-300 font-medium">Communities Found: {analyticsData.communitiesCount || 0}</div>
                      <div className="text-green-400/70">Documents grouped by similarity</div>
                    </div>
                  </div>
                )}
                
                <div className="grid grid-cols-2 gap-2 text-sm">
                  <div className="bg-slate-700/30 p-2 rounded">
                    <div className="text-slate-400 text-xs">Documents</div>
                    <div className="font-medium text-blue-400">{graphStats?.total_stats?.total_nodes || graphStats?.node_types?.document || 0}</div>
                  </div>
                  <div className="bg-slate-700/30 p-2 rounded">
                    <div className="text-slate-400 text-xs">Communities</div>
                    <div className="font-medium text-green-400">{analyticsData.communitiesCount || 0}</div>
                  </div>
                </div>
              </>
            ) : (
              <div className="bg-slate-700/30 p-4 rounded text-center">
                <div className="text-slate-400 text-xs">No analytics data available</div>
              </div>
            )}
          </div>

          {/* Graph Tips */}
          <div className="p-4">
            <div className="bg-slate-700/30 border border-slate-600/50 rounded-lg p-4">
              <h4 className="text-sm font-medium text-slate-200 mb-2 flex items-center gap-2">
                <div className="w-2 h-2 bg-blue-400 rounded-full"></div>
                Interaction Tips
              </h4>
              <div className="space-y-2 text-xs text-slate-400">
                <div className="flex items-center gap-2">
                  <span className="text-blue-400">•</span>
                  <span>Click links to see relationships</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-yellow-400">•</span>
                  <span>Switch analytics modes above</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-purple-400">•</span>
                  <span>Use zoom and pan to explore</span>
                </div>
              </div>
            </div>
          </div>
          </div>
        </div>
      </div>
    </div>
  );
}