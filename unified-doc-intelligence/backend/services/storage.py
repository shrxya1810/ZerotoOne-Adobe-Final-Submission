"""
Storage service for sessions, documents, and search indexing.
Combines SQLite for metadata and FAISS for vector search.
"""
import os
import json
# Try to import sqlite3, fallback to pysqlite3 if built-in is missing
try:
    import sqlite3
except ImportError:
    try:
        import pysqlite3.dbapi2 as sqlite3
        print("✅ Using pysqlite3 as SQLite backend")
    except ImportError:
        print("❌ No SQLite implementation available. Please install pysqlite3-binary")
        raise ImportError("SQLite support not available")
import time
import uuid
import logging
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime

from langchain.schema import Document

from ..settings import settings
from ..models.schemas import Session, DocumentMetadata, DocumentInfo

logger = logging.getLogger(__name__)


class StorageService:
    """
    Unified storage service for sessions, documents, and search indices.
    """
    
    def __init__(self):
        self.db_path = settings.SQLITE_DB
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database with required tables."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            # Sessions table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    name TEXT,
                    created_at INTEGER,
                    last_accessed INTEGER,
                    expires_at INTEGER
                )
            ''')
            
            # Documents table
            conn.execute('''
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    session_id TEXT,
                    filename TEXT,
                    file_size INTEGER,
                    upload_time INTEGER,
                    page_count INTEGER,
                    title TEXT,
                    author TEXT,
                    doc_type TEXT,
                    status TEXT,
                    error_message TEXT,
                    metadata_json TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id)
                )
            ''')
            
            # Chunks table for search
            conn.execute('''
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    doc_id TEXT,
                    session_id TEXT,
                    content TEXT,
                    page_number INTEGER,
                    chunk_index INTEGER,
                    metadata_json TEXT,
                    FOREIGN KEY (doc_id) REFERENCES documents (doc_id),
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id)
                )
            ''')
            
            # FTS5 table for keyword search
            conn.execute('''
                CREATE VIRTUAL TABLE IF NOT EXISTS fts_chunks
                USING fts5(
                    chunk_id,
                    doc_id UNINDEXED,
                    session_id UNINDEXED,
                    page_number UNINDEXED,
                    content,
                    tokenize = 'porter'
                )
            ''')
            
            # Create indices
            conn.execute('CREATE INDEX IF NOT EXISTS idx_documents_session ON documents(session_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_chunks_session ON chunks(session_id)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id)')
            
            conn.commit()
        
        logger.info("Database initialized successfully")
    
    def create_session(self, name: Optional[str] = None) -> Session:
        """Create a new session."""
        session_id = uuid.uuid4().hex
        now = int(time.time())
        expires_at = now + (settings.SESSION_TIMEOUT_HOURS * 3600)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                'INSERT INTO sessions (session_id, name, created_at, last_accessed, expires_at) VALUES (?, ?, ?, ?, ?)',
                (session_id, name, now, now, expires_at)
            )
            conn.commit()
        
        session = Session(
            session_id=session_id,
            name=name,
            created_at=datetime.fromtimestamp(now),
            last_accessed=datetime.fromtimestamp(now),
            document_count=0
        )
        
        logger.info(f"Created session {session_id}")
        return session
    
    def get_session(self, session_id: str) -> Optional[Session]:
        """Get session by ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT s.session_id, s.name, s.created_at, s.last_accessed, 
                       COUNT(d.doc_id) as doc_count
                FROM sessions s
                LEFT JOIN documents d ON s.session_id = d.session_id
                WHERE s.session_id = ? AND s.expires_at > ?
                GROUP BY s.session_id, s.name, s.created_at, s.last_accessed
                ''', (session_id, int(time.time())))
            
            row = cursor.fetchone()
            
            if not row:
                return None
            
            # Update last accessed time
            conn.execute(
                "UPDATE sessions SET last_accessed = ? WHERE session_id = ?",
                (int(time.time()), session_id)
            )
            conn.commit()
            
            return Session(
                session_id=row[0],
                name=row[1],
                created_at=datetime.fromtimestamp(row[2]),
                last_accessed=datetime.fromtimestamp(row[3]),
                document_count=row[4]
            )
    
    def cleanup_expired_sessions(self):
        """Clean up expired sessions and their documents."""
        now = int(time.time())
        
        with sqlite3.connect(self.db_path) as conn:
            # Get expired sessions
            cursor = conn.execute(
                "SELECT session_id FROM sessions WHERE expires_at <= ?",
                (now,)
            )
            expired_sessions = [row[0] for row in cursor.fetchall()]
            
            for session_id in expired_sessions:
                self._delete_session_data(conn, session_id)
            
            conn.commit()
        
        logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
    
    def _delete_session_data(self, conn: sqlite3.Connection, session_id: str):
        """Delete all data for a session."""
        # Delete from FTS
        conn.execute("DELETE FROM fts_chunks WHERE session_id = ?", (session_id,))
        
        # Delete chunks
        conn.execute("DELETE FROM chunks WHERE session_id = ?", (session_id,))
        
        # Delete documents
        conn.execute("DELETE FROM documents WHERE session_id = ?", (session_id,))
        
        # Delete session
        conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
    
    def add_document(self, session_id: str, filename: str, file_size: int, doc_type: str) -> str:
        """Add a document record."""
        doc_id = uuid.uuid4().hex
        now = int(time.time())
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO documents 
                (doc_id, session_id, filename, file_size, upload_time, doc_type, status)
                VALUES (?, ?, ?, ?, ?, ?, 'uploaded')
                ''', (doc_id, session_id, filename, file_size, now, doc_type))
            conn.commit()
        
        logger.info(f"Added document {filename} to session {session_id}")
        return doc_id
    
    def update_document_status(self, doc_id: str, status: str, 
                             error_message: Optional[str] = None,
                             metadata: Optional[Dict] = None):
        """Update document processing status."""
        with sqlite3.connect(self.db_path) as conn:
            if metadata:
                conn.execute('''
                    UPDATE documents 
                    SET status = ?, error_message = ?, page_count = ?, title = ?, 
                        author = ?, metadata_json = ?
                    WHERE doc_id = ?
                    ''', (status, error_message, metadata.get('page_count'),
                         metadata.get('title'), metadata.get('author'),
                         json.dumps(metadata), doc_id))
            else:
                conn.execute('''
                    UPDATE documents 
                    SET status = ?, error_message = ?
                    WHERE doc_id = ?
                    ''', (status, error_message, doc_id))
            conn.commit()
    
    def get_session_documents(self, session_id: str) -> List[Dict]:
        """Get all documents in a session."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT doc_id, filename, file_size, upload_time, page_count, 
                       title, author, doc_type, status, error_message, metadata_json
                FROM documents 
                WHERE session_id = ?
                ORDER BY upload_time DESC
                ''', (session_id,))
            
            documents = []
            for row in cursor.fetchall():
                documents.append({
                    'document_id': row[0],
                    'filename': row[1],
                    'file_size': row[2],
                    'upload_time': row[3],
                    'page_count': row[4],
                    'title': row[5],
                    'author': row[6],
                    'doc_type': row[7],
                    'status': row[8],
                    'error_message': row[9],
                    'metadata_json': row[10]
                })
            
            return documents
    
    def add_chunks(self, doc_id: str, session_id: str, chunks: List[Document]):
        """Add document chunks for search indexing."""
        with sqlite3.connect(self.db_path) as conn:
            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc_id}_chunk_{i}"
                page_number = chunk.metadata.get('page_number', 0)
                
                # Add to chunks table
                conn.execute('''
                    INSERT INTO chunks 
                    (chunk_id, doc_id, session_id, content, page_number, chunk_index, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (chunk_id, doc_id, session_id, chunk.page_content, 
                         page_number, i, json.dumps(chunk.metadata)))
                
                # Add to FTS
                conn.execute('''
                    INSERT INTO fts_chunks (chunk_id, doc_id, session_id, page_number, content)
                    VALUES (?, ?, ?, ?, ?)
                    ''', (chunk_id, doc_id, session_id, page_number, chunk.page_content))
            
            conn.commit()
        
        logger.info(f"Added {len(chunks)} chunks for document {doc_id}")
    
    def keyword_search(self, session_id: str, query: str, top_k: int = 10) -> List[Dict]:
        """Perform keyword search using FTS5."""
        logger.info(f"🔍 Executing FTS search: session_id='{session_id}', query='{query}', top_k={top_k}")
        
        with sqlite3.connect(self.db_path) as conn:
            # First check if FTS table has data for this session
            cursor = conn.execute('''
                SELECT COUNT(*) FROM fts_chunks WHERE session_id = ?
                ''', (session_id,))
            
            fts_count = cursor.fetchone()[0]
            logger.info(f"📊 FTS table has {fts_count} chunks for session {session_id}")
            
            if fts_count == 0:
                logger.warning(f"⚠️ No FTS data found for session {session_id}")
                return []
            
            # Perform the actual search
            cursor = conn.execute('''
                SELECT f.chunk_id, f.doc_id, f.page_number, f.content,
                       d.filename, bm25(fts_chunks) as score
                FROM fts_chunks f
                JOIN documents d ON f.doc_id = d.doc_id
                WHERE fts_chunks MATCH ? AND f.session_id = ?
                ORDER BY bm25(fts_chunks)
                LIMIT ?
                ''', (query, session_id, top_k))
            
            results = []
            raw_scores = []
            
            # First pass: collect all rows and scores
            rows = cursor.fetchall()
            for row in rows:
                raw_scores.append(float(row[5]))
            
            # Normalize scores to 0-1 range
            if raw_scores:
                min_score = min(raw_scores)
                max_score = max(raw_scores)
                score_range = max_score - min_score
                
                # Handle case where all scores are the same
                if score_range == 0:
                    normalized_scores = [1.0] * len(raw_scores)
                else:
                    # Normalize to 0-1 range
                    normalized_scores = [(score - min_score) / score_range for score in raw_scores]
            else:
                normalized_scores = []
            
            # Second pass: create results with normalized scores
            for i, row in enumerate(rows):
                results.append({
                    'chunk_id': row[0],
                    'doc_id': row[1],
                    'document': row[4],
                    'content': row[3],
                    'page_number': row[2],
                    'score': normalized_scores[i] if normalized_scores else 0.0
                })
            
            logger.info(f"✅ FTS search completed: {len(results)} results found")
            return results
    
    def get_session_chunks(self, session_id: str) -> List[Document]:
        """Get all chunks for a session (for vector search)."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT chunk_id, content, page_number, metadata_json
                FROM chunks
                WHERE session_id = ?
                ORDER BY doc_id, chunk_index
                ''', (session_id,))
            
            chunks = []
            for row in cursor.fetchall():
                metadata = json.loads(row[3]) if row[3] else {}
                metadata['chunk_id'] = row[0]
                metadata['page_number'] = row[2]
                
                chunks.append(Document(
                    page_content=row[1],
                    metadata=metadata
                ))
            
            return chunks
    
    def get_document_chunks(self, doc_id: str) -> List[Dict]:
        """Get all chunks for a specific document."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT chunk_id, content, page_number, chunk_index, metadata_json
                FROM chunks
                WHERE doc_id = ?
                ORDER BY chunk_index
                ''', (doc_id,))
            
            chunks = []
            for row in cursor.fetchall():
                metadata = json.loads(row[4]) if row[4] else {}
                chunks.append({
                    'chunk_id': row[0],
                    'content': row[1],
                    'page_number': row[2],
                    'chunk_index': row[3],
                    'metadata': metadata
                })
            
            return chunks


# Global storage service instance
storage_service = StorageService()


# Convenience functions
def create_session(name: Optional[str] = None) -> Session:
    """Create a new session."""
    return storage_service.create_session(name)


def get_session(session_id: str) -> Optional[Session]:
    """Get session by ID."""
    return storage_service.get_session(session_id)


def add_document(session_id: str, filename: str, file_size: int, doc_type: str) -> str:
    """Add a document to a session."""
    return storage_service.add_document(session_id, filename, file_size, doc_type)


def update_document_status(doc_id: str, status: str, error_message: Optional[str] = None,
                          metadata: Optional[Dict] = None):
    """Update document status."""
    storage_service.update_document_status(doc_id, status, error_message, metadata)


def get_session_documents(session_id: str) -> List[DocumentInfo]:
    """Get documents in a session."""
    return storage_service.get_session_documents(session_id)


def add_chunks(doc_id: str, session_id: str, chunks: List[Document]):
    """Add chunks for search indexing."""
    storage_service.add_chunks(doc_id, session_id, chunks)


def keyword_search(session_id: str, query: str, top_k: int = 10) -> List[Dict]:
    """Perform keyword search."""
    return storage_service.keyword_search(session_id, query, top_k)


def get_document_chunks(doc_id: str) -> List[Dict]:
    """Get chunks for a specific document."""
    return storage_service.get_document_chunks(doc_id)
