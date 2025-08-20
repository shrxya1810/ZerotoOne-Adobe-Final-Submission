"""
Session management router.
"""
from fastapi import APIRouter, HTTPException, Depends
from typing import Optional

from ..services.storage import storage_service
from ..models.schemas import SessionCreate, SessionResponse, Session

router = APIRouter()


@router.post("/create", response_model=SessionResponse)
async def create_session(request: Optional[SessionCreate] = None):
    """
    Create a new document session.
    
    A session groups related documents together for search and analysis.
    Sessions automatically expire after the configured timeout period.
    
    Args:
        request: Optional session creation parameters
        
    Returns:
        SessionResponse with the new session details
    """
    try:
        session_name = None
        if request:
            session_name = request.name
        
        session = storage_service.create_session(session_name)
        
        return SessionResponse(
            session=session,
            message=f"Session created successfully: {session.session_id}"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create session: {str(e)}"
        )


@router.get("/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str):
    """
    Get session information by ID.
    
    Args:
        session_id: Session identifier
        
    Returns:
        SessionResponse with session details
    """
    session = storage_service.get_session(session_id)
    
    if not session:
        raise HTTPException(
            status_code=404,
            detail=f"Session not found: {session_id}"
        )
    
    return SessionResponse(
        session=session,
        message="Session retrieved successfully"
    )


@router.delete("/{session_id}")
async def delete_session(session_id: str):
    """
    Delete a session and all its associated data.
    
    This will permanently remove:
    - All documents in the session
    - All search indices
    - All analysis results
    
    Args:
        session_id: Session identifier
        
    Returns:
        Success confirmation
    """
    session = storage_service.get_session(session_id)
    
    if not session:
        raise HTTPException(
            status_code=404,
            detail=f"Session not found: {session_id}"
        )
    
    try:
        # Delete session data
        import sqlite3
        with sqlite3.connect(storage_service.db_path) as conn:
            storage_service._delete_session_data(conn, session_id)
            conn.commit()
        
        return {
            "success": True,
            "message": f"Session {session_id} deleted successfully"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete session: {str(e)}"
        )


@router.post("/{session_id}/extend")
async def extend_session(session_id: str, hours: int = 24):
    """
    Extend a session's expiration time.
    
    Args:
        session_id: Session identifier
        hours: Additional hours to extend (default: 24)
        
    Returns:
        Updated session information
    """
    session = storage_service.get_session(session_id)
    
    if not session:
        raise HTTPException(
            status_code=404,
            detail=f"Session not found: {session_id}"
        )
    
    try:
        import sqlite3
        import time
        
        additional_seconds = hours * 3600
        
        with sqlite3.connect(storage_service.db_path) as conn:
            conn.execute(
                "UPDATE sessions SET expires_at = expires_at + ? WHERE session_id = ?",
                (additional_seconds, session_id)
            )
            conn.commit()
        
        # Get updated session
        updated_session = storage_service.get_session(session_id)
        
        return SessionResponse(
            session=updated_session,
            message=f"Session extended by {hours} hours"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to extend session: {str(e)}"
        )


@router.post("/cleanup")
async def cleanup_expired_sessions():
    """
    Manually trigger cleanup of expired sessions.
    
    This endpoint is useful for maintenance or testing.
    Normally, cleanup happens automatically during startup.
    
    Returns:
        Cleanup summary
    """
    try:
        # Get count before cleanup
        import sqlite3
        import time
        
        with sqlite3.connect(storage_service.db_path) as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM sessions WHERE expires_at <= ?",
                (int(time.time()),)
            )
            expired_count = cursor.fetchone()[0]
        
        # Perform cleanup
        storage_service.cleanup_expired_sessions()
        
        return {
            "success": True,
            "message": f"Cleaned up {expired_count} expired sessions",
            "expired_sessions_removed": expired_count
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Cleanup failed: {str(e)}"
        )
