"""
PDF streaming and file serving router.
"""
import os
from fastapi import APIRouter, HTTPException, Request, Header
from fastapi.responses import FileResponse, StreamingResponse
from typing import Optional, List
import logging

from ..settings import settings
from ..services.storage import storage_service
from ..models.schemas import DocumentListResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/{filename}")
async def serve_pdf(
    filename: str,
    request: Request,
    range: Optional[str] = Header(None)
):
    """
    Serve PDF files with support for range requests (for streaming).
    
    This endpoint serves uploaded PDF files with proper CORS headers
    and support for HTTP range requests, enabling:
    - Progressive loading in PDF viewers
    - Seeking to specific pages
    - Bandwidth-efficient streaming
    
    Args:
        filename: PDF filename to serve
        request: FastAPI request object
        range: HTTP Range header for partial content requests
        
    Returns:
        FileResponse or StreamingResponse with PDF content
    """
    # Construct file path
    file_path = os.path.join(settings.UPLOAD_DIR, filename)
    
    # Check if file exists
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="PDF file not found")
    
    # Get file size
    file_size = os.path.getsize(file_path)
    
    # Handle range requests for streaming
    if range:
        try:
            # Parse range header (e.g., "bytes=0-1023")
            range_match = range.replace('bytes=', '').split('-')
            start = int(range_match[0]) if range_match[0] else 0
            end = int(range_match[1]) if range_match[1] else file_size - 1
            
            # Validate range
            if start >= file_size or end >= file_size or start > end:
                raise HTTPException(
                    status_code=416, 
                    detail="Requested range not satisfiable"
                )
            
            # Calculate content length
            content_length = end - start + 1
            
            def generate_file_chunk():
                with open(file_path, 'rb') as f:
                    f.seek(start)
                    remaining = content_length
                    chunk_size = settings.CHUNK_SIZE  # Use configured chunk size
                    
                    while remaining > 0:
                        chunk = f.read(min(chunk_size, remaining))
                        if not chunk:
                            break
                        remaining -= len(chunk)
                        yield chunk
            
            headers = {
                'Content-Range': f'bytes {start}-{end}/{file_size}',
                'Accept-Ranges': 'bytes',
                'Content-Length': str(content_length),
                'Content-Type': 'application/pdf',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'GET, HEAD, OPTIONS',
                'Access-Control-Allow-Headers': 'Range, Content-Range',
                'Access-Control-Expose-Headers': 'Content-Range, Accept-Ranges, Content-Length'
            }
            
            return StreamingResponse(
                generate_file_chunk(),
                status_code=206,  # Partial Content
                headers=headers,
                media_type='application/pdf'
            )
            
        except ValueError:
            # Invalid range format, fall back to full file
            pass
    
    # Serve full file
    headers = {
        'Accept-Ranges': 'bytes',
        'Content-Length': str(file_size),
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET, HEAD, OPTIONS',
        'Access-Control-Allow-Headers': 'Range, Content-Range',
        'Access-Control-Expose-Headers': 'Content-Range, Accept-Ranges, Content-Length'
    }
    
    return FileResponse(
        file_path,
        media_type='application/pdf',
        headers=headers,
        filename=filename
    )


@router.get("/{session_id}/{filename}")
async def serve_session_pdf(
    session_id: str,
    filename: str,
    request: Request,
    range: Optional[str] = Header(None)
):
    """
    Serve PDF files for a specific session with support for range requests.
    
    This endpoint serves uploaded PDF files associated with a session,
    with proper CORS headers and support for HTTP range requests.
    
    Args:
        session_id: Session identifier
        filename: PDF filename to serve
        request: FastAPI request object
        range: HTTP Range header for partial content requests
        
    Returns:
        FileResponse or StreamingResponse with PDF content
    """
    # Verify session exists
    session = storage_service.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=404,
            detail=f"Session not found: {session_id}"
        )
    
    # Find the actual file with doc_id prefix
    # Files are stored as {doc_id}_{filename}
    upload_dir = settings.UPLOAD_DIR
    actual_filename = None
    
    # Search for file with this session and filename
    if os.path.exists(upload_dir):
        for file in os.listdir(upload_dir):
            if file.endswith(f"_{filename}"):
                # Found a file with matching filename suffix
                actual_filename = file
                break
    
    if not actual_filename:
        raise HTTPException(status_code=404, detail=f"PDF file '{filename}' not found in session {session_id}")
    
    file_path = os.path.join(upload_dir, actual_filename)
    
    # Check if file exists
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="PDF file not found")
    
    # Get file size
    file_size = os.path.getsize(file_path)
    
    # Handle range requests for efficient streaming
    if range:
        range_match = range.replace('bytes=', '').split('-')
        range_start = int(range_match[0]) if range_match[0] else 0
        range_end = int(range_match[1]) if range_match[1] else file_size - 1
        
        if range_start >= file_size:
            raise HTTPException(status_code=416, detail="Range not satisfiable")
        
        range_end = min(range_end, file_size - 1)
        content_length = range_end - range_start + 1
        
        def iter_file():
            with open(file_path, 'rb') as f:
                f.seek(range_start)
                remaining = content_length
                while remaining > 0:
                    chunk_size = min(settings.CHUNK_SIZE, remaining)
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    remaining -= len(chunk)
                    yield chunk
        
        headers = {
            'Content-Range': f'bytes {range_start}-{range_end}/{file_size}',
            'Accept-Ranges': 'bytes',
            'Content-Length': str(content_length),
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET, HEAD, OPTIONS',
            'Access-Control-Allow-Headers': 'Range, Content-Range',
            'Access-Control-Expose-Headers': 'Content-Range, Accept-Ranges, Content-Length'
        }
        
        return StreamingResponse(
            iter_file(),
            status_code=206,
            headers=headers,
            media_type='application/pdf'
        )
    
    # Return full file for non-range requests
    headers = {
        'Content-Length': str(file_size),
        'Accept-Ranges': 'bytes',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET, HEAD, OPTIONS',
        'Access-Control-Allow-Headers': 'Range, Content-Range',
        'Access-Control-Expose-Headers': 'Content-Range, Accept-Ranges, Content-Length'
    }
    
    return FileResponse(
        file_path,
        media_type='application/pdf',
        headers=headers,
        filename=filename
    )


@router.get("/list/{session_id}", response_model=DocumentListResponse)
async def list_session_documents(session_id: str):
    """
    List all documents in a session.
    
    Args:
        session_id: Session identifier
        
    Returns:
        List of documents with their metadata and status
    """
    # Verify session exists
    session = storage_service.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=404,
            detail=f"Session not found: {session_id}"
        )
    
    try:
        documents = storage_service.get_session_documents(session_id)
        
        return DocumentListResponse(
            session_id=session_id,
            documents=documents,
            total_count=len(documents),
            message=f"Found {len(documents)} documents in session"
        )
        
    except Exception as e:
        logger.error(f"Error listing documents for session {session_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list documents: {str(e)}"
        )


@router.head("/{filename}")
async def head_pdf(filename: str):
    """
    HEAD request for PDF files (for checking file existence and size).
    
    Args:
        filename: PDF filename
        
    Returns:
        Response headers without body
    """
    file_path = os.path.join(settings.UPLOAD_DIR, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="PDF file not found")
    
    file_size = os.path.getsize(file_path)
    
    headers = {
        'Accept-Ranges': 'bytes',
        'Content-Length': str(file_size),
        'Content-Type': 'application/pdf',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET, HEAD, OPTIONS',
        'Access-Control-Allow-Headers': 'Range, Content-Range',
        'Access-Control-Expose-Headers': 'Content-Range, Accept-Ranges, Content-Length'
    }
    
    return FileResponse(
        file_path,
        media_type='application/pdf',
        headers=headers,
        filename=filename
    )


@router.options("/{filename}")
async def options_pdf(filename: str):
    """
    OPTIONS request for CORS preflight.
    
    Args:
        filename: PDF filename (for route matching)
        
    Returns:
        CORS headers
    """
    headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET, HEAD, OPTIONS',
        'Access-Control-Allow-Headers': 'Range, Content-Range, Content-Type',
        'Access-Control-Expose-Headers': 'Content-Range, Accept-Ranges, Content-Length',
        'Access-Control-Max-Age': '86400'  # 24 hours
    }
    
    return {"message": "CORS preflight"}, 200, headers


@router.get("/")
async def list_all_pdfs():
    """
    List all available PDF files (for debugging/admin).
    
    Returns:
        List of available PDF files
    """
    try:
        upload_dir = settings.UPLOAD_DIR
        
        if not os.path.exists(upload_dir):
            return {"files": [], "total": 0}
        
        pdf_files = []
        for filename in os.listdir(upload_dir):
            if filename.lower().endswith('.pdf'):
                file_path = os.path.join(upload_dir, filename)
                file_size = os.path.getsize(file_path)
                pdf_files.append({
                    "filename": filename,
                    "size": file_size,
                    "size_mb": round(file_size / 1024 / 1024, 2)
                })
        
        return {
            "files": pdf_files,
            "total": len(pdf_files),
            "upload_dir": upload_dir
        }
        
    except Exception as e:
        logger.error(f"Error listing PDF files: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list files: {str(e)}"
        )
