"""
File upload router with document processing.
"""
import os
import tempfile
import uuid
import asyncio
import time
from typing import List
from concurrent.futures import ThreadPoolExecutor
from fastapi import APIRouter, File, UploadFile, HTTPException, Form, BackgroundTasks
import logging

from ..settings import settings
from ..services.storage import storage_service, add_document, update_document_status, add_chunks
from ..services.embed import chunk_documents
from ..services.index import add_documents_to_index
from ..models.schemas import UploadResponse, DocumentInfo, DocumentType

logger = logging.getLogger(__name__)

router = APIRouter()


def process_document_sync(doc_id: str, session_id: str, file_path: str, filename: str):
    """
    Synchronous document processing function for concurrent execution.
    """
    try:
        # Update status to processing
        update_document_status(doc_id, "processing")
        
        # Extract text content based on file type
        if filename.lower().endswith('.pdf'):
            # Use optimized PDF extraction service
            from ..services.pdf_extract import pdf_extract_service
            extract_response = pdf_extract_service.extract_pdf_structure(file_path)
            
            if not extract_response.success:
                raise ValueError(f"PDF extraction failed: {extract_response.message}")
            
            # Get full text content
            content = extract_pdf_content_sync(file_path)
        else:
            raise ValueError(f"Unsupported file type: {filename}")
        
        # Create document object for chunking
        from langchain.schema import Document
        document = Document(
            page_content=content,
            metadata={
                "filename": filename,
                "doc_id": doc_id,
                "source": filename,
                "title": extract_response.title if 'extract_response' in locals() else None,
                "page_count": extract_response.page_count if 'extract_response' in locals() else 0
            }
        )
        
        # Chunk the document
        chunks = chunk_documents([document])
        
        # Add proper page numbers to chunks based on actual page markers in content
        actual_page_count = extract_response.page_count if 'extract_response' in locals() else 0
        
        for i, chunk in enumerate(chunks):
            # Try to extract page number from content markers
            chunk_content = chunk.page_content
            estimated_page = 0  # Default to page 0 (0-indexed)
            
            # Look for page markers in the chunk content
            import re
            page_matches = re.findall(r'--- Page (\d+) ---', chunk_content)
            if page_matches:
                # Use the last page marker found in the chunk
                page_number_from_content = int(page_matches[-1])
                estimated_page = max(0, page_number_from_content - 1)  # Convert to 0-indexed
            else:
                # Fallback to improved distribution-based estimation
                if actual_page_count > 0:
                    # Distribute chunks evenly across actual pages
                    estimated_page = min(i * actual_page_count // len(chunks), actual_page_count - 1)
                else:
                    # Fallback to conservative estimation
                    estimated_page = min(i // 3, 50)  # More conservative estimate, max 50 pages
            
            chunk.metadata["page_number"] = estimated_page
        
        # Store chunks in database
        add_chunks(doc_id, session_id, chunks)
        
        # Add to vector index
        success = add_documents_to_index(session_id, chunks)
        
        if success:
            # Update status to indexed
            metadata = {
                "page_count": extract_response.page_count if 'extract_response' in locals() else len(chunks) // 2,
                "chunk_count": len(chunks),
                "title": extract_response.title if 'extract_response' in locals() else None
            }
            update_document_status(doc_id, "indexed", metadata=metadata)
        else:
            update_document_status(doc_id, "error", "Failed to index document")
        
    except Exception as e:
        logger.error(f"Error processing document {doc_id}: {e}")
        update_document_status(doc_id, "error", str(e))
    
    finally:
        # Keep the file for serving, don't delete it
        # The file will be served by the PDF router
        pass


async def process_documents_concurrently(doc_tasks: List[tuple]):
    """
    Process multiple documents concurrently using ThreadPoolExecutor.
    
    Args:
        doc_tasks: List of tuples (doc_id, session_id, file_path, filename)
    """
    if not doc_tasks:
        return
    
    start_time = time.time()
    
    # Limit concurrent workers to prevent resource exhaustion
    max_workers = min(len(doc_tasks), settings.MAX_CONCURRENT_UPLOADS)
    
    logger.info(f"Starting concurrent processing of {len(doc_tasks)} documents with {max_workers} workers")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = [
            asyncio.get_event_loop().run_in_executor(
                executor, process_document_sync, *task_args
            ) for task_args in doc_tasks
        ]
        
        # Wait for all tasks to complete, capturing exceptions
        results = await asyncio.gather(*futures, return_exceptions=True)
        
        # Log results and performance
        successful_count = 0
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                doc_id = doc_tasks[i][0]
                logger.error(f"Document processing failed for {doc_id}: {result}")
            else:
                successful_count += 1
        
        total_time = time.time() - start_time
        logger.info(f"Completed concurrent processing: {successful_count}/{len(doc_tasks)} successful in {total_time:.2f}s")


def extract_pdf_content_sync(file_path: str) -> str:
    """Extract text content from PDF synchronously using optimized parallel extraction."""
    try:
        import fitz  # PyMuPDF
        from ..services.pdf_extract import pdf_extract_service
        
        doc = fitz.open(file_path)
        
        # Use optimized parallel extraction for better performance
        content = pdf_extract_service.extract_content_parallel(doc)
        
        doc.close()
        return content
        
    except Exception as e:
        logger.error(f"Error extracting PDF content: {e}")
        raise


async def extract_pdf_content(file_path: str) -> str:
    """Extract text content from PDF."""
    try:
        import fitz  # PyMuPDF
        
        doc = fitz.open(file_path)
        content = ""
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            content += page.get_text()
            content += f"\n\n--- Page {page_num + 1} ---\n\n"
        
        doc.close()
        return content
        
    except Exception as e:
        logger.error(f"Error extracting PDF content: {e}")
        raise


@router.post("/batch", response_model=UploadResponse)
async def upload_files(
    background_tasks: BackgroundTasks,
    session_id: str = Form(...),
    files: List[UploadFile] = File(...)
):
    """
    Upload multiple files to a session for processing.
    
    This endpoint:
    1. Validates the session and files
    2. Saves files to storage
    3. Starts background processing for text extraction and indexing
    4. Returns immediate response with upload status
    
    Args:
        session_id: Target session ID
        files: List of files to upload (PDFs supported)
        
    Returns:
        UploadResponse with upload status for each file
    """
    # Validate session
    session = storage_service.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=404,
            detail=f"Session not found: {session_id}"
        )
    
    if not files:
        raise HTTPException(
            status_code=400,
            detail="No files provided"
        )
    
    # Validate file count
    if len(files) > settings.MAX_FILES_PER_SESSION:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files. Maximum {settings.MAX_FILES_PER_SESSION} files per session"
        )
    
    uploaded_files = []
    doc_tasks = []  # Collect tasks for concurrent processing
    
    for file in files:
        try:
            # Validate file
            if not file.filename:
                uploaded_files.append(DocumentInfo(
                    filename="unknown",
                    status="error",
                    error_message="No filename provided"
                ))
                continue
            
            # Check file size
            content = await file.read()
            file_size = len(content)
            
            if file_size > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
                uploaded_files.append(DocumentInfo(
                    filename=file.filename,
                    status="error",
                    error_message=f"File too large. Maximum {settings.MAX_FILE_SIZE_MB}MB"
                ))
                continue
            
            # Determine file type
            doc_type = DocumentType.PDF if file.filename.lower().endswith('.pdf') else DocumentType.TXT
            
            # Create document record
            doc_id = add_document(session_id, file.filename, file_size, doc_type.value)
            
            # Save file to storage
            file_path = os.path.join(settings.UPLOAD_DIR, f"{doc_id}_{file.filename}")
            with open(file_path, 'wb') as f:
                f.write(content)
            
            # Collect task for concurrent processing
            doc_tasks.append((doc_id, session_id, file_path, file.filename))
            
            uploaded_files.append(DocumentInfo(
                filename=file.filename,
                status="uploaded"
            ))
            
        except Exception as e:
            logger.error(f"Error uploading file {file.filename}: {e}")
            uploaded_files.append(DocumentInfo(
                filename=file.filename or "unknown",
                status="error",
                error_message=str(e)
            ))
    
    # Start concurrent processing for all successfully uploaded files
    if doc_tasks:
        background_tasks.add_task(process_documents_concurrently, doc_tasks)
    
    return UploadResponse(
        uploaded_files=uploaded_files,
        session_id=session_id,
        message=f"Uploaded {len([f for f in uploaded_files if f.status == 'uploaded'])} files successfully. Processing started concurrently."
    )


@router.get("/status/{session_id}")
async def get_upload_status(session_id: str):
    """
    Get upload and processing status for all documents in a session.
    
    Args:
        session_id: Session identifier
        
    Returns:
        Status summary for all documents
    """
    # Validate session
    session = storage_service.get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=404,
            detail=f"Session not found: {session_id}"
        )
    
    try:
        documents = storage_service.get_session_documents(session_id)
        
        status_summary = {
            "session_id": session_id,
            "total_documents": len(documents),
            "status_counts": {},
            "documents": []
        }
        
        for doc in documents:
            # Count statuses
            if doc.status not in status_summary["status_counts"]:
                status_summary["status_counts"][doc.status] = 0
            status_summary["status_counts"][doc.status] += 1
            
            # Add document info
            status_summary["documents"].append({
                "filename": doc.filename,
                "status": doc.status,
                "error_message": doc.error_message
            })
        
        return status_summary
        
    except Exception as e:
        logger.error(f"Error getting upload status for session {session_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get status: {str(e)}"
        )


# Include this router in the main app with prefix="/upload"
