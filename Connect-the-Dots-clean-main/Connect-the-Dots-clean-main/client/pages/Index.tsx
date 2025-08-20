import { useEffect } from 'react';
import { useAppStore } from '../lib/store';
import SimplifiedUploadPage from './SimplifiedUploadPage';
import WorkspacePage from './WorkspacePage';
import KnowledgeGraphPage from './KnowledgeGraphPage';

export default function Index() {
  const { currentView, documents } = useAppStore();

  // If we have documents and current view is workspace, show workspace
  if (documents.length > 0 && currentView === 'workspace') {
    return <WorkspacePage />;
  }

  // If current view is knowledge-graph, show knowledge graph
  if (currentView === 'knowledge-graph') {
    return <KnowledgeGraphPage />;
  }

  // Default to upload page
  return <SimplifiedUploadPage />;
}
