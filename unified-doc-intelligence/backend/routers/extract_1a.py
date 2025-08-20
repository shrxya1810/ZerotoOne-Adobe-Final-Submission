"""
PDF extraction router for Challenge 1a - PDF title and heading extraction.
"""
import os
import time
import tempfile
import uuid
import asyncio
import aiofiles
from concurrent.futures import ThreadPoolExecutor
from typing import List
from fastapi import APIRouter, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from ..services.pdf_extract import pdf_extract_service
from ..models.schemas import ExtractResponse
from ..settings import settings

router = APIRouter()


async def save_file_async(content: bytes, temp_path: str) -> None:
    """Save file content asynchronously."""
    async with aiofiles.open(temp_path, 'wb') as temp_file:
        await temp_file.write(content)


def extract_pdf_sync(temp_path: str) -> ExtractResponse:
    """Synchronous PDF extraction for thread pool execution."""
    return pdf_extract_service.extract_pdf_structure(temp_path)


@router.post("/process-pdf", response_model=ExtractResponse)
async def process_pdf(file: UploadFile = File(...)):
    """
    Extract title and hierarchical outline from a PDF file with optimized async processing.
    
    This endpoint accepts a PDF file and returns:
    - Document title (if extractable)
    - Hierarchical outline with headings, levels, and page numbers (0-indexed)
    - Processing metadata
    
    The page numbers returned are 0-indexed to match frontend PDF viewers.
    Each outline item includes:
    - level: Heading level (1=H1, 2=H2, etc.)
    - title: Heading text (cleaned)
    - page_number: 0-indexed page number for navigation
    
    Optimizations:
    - Async file I/O for better concurrency
    - Threaded PDF processing for CPU-intensive operations
    - Parallel page processing within PDF extraction
    """
    start_time = time.time()
    
    # Validate file type
    if not file.filename or not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400, 
            detail="Invalid file type. Please upload a PDF file."
        )
    
    # Create temporary file
    temp_dir = tempfile.gettempdir()
    temp_filename = f"{uuid.uuid4()}_{file.filename}"
    temp_path = os.path.join(temp_dir, temp_filename)
    
    try:
        # Async file operations for better concurrency
        content = await file.read()
        await save_file_async(content, temp_path)
        
        file_save_time = time.time() - start_time
        
        # Extract structure using thread pool for CPU-intensive work
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, extract_pdf_sync, temp_path
        )
        
        total_time = time.time() - start_time
        
        # Add timing information to result
        if hasattr(result, 'processing_time'):
            result.processing_time = total_time
        
        # Log performance metrics
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"⚡ PDF extraction completed: {total_time:.2f}s total "
                   f"(file save: {file_save_time:.2f}s, "
                   f"extraction: {result.processing_time:.2f}s)")
        
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"PDF processing failed: {str(e)}"
        )
    
    finally:
        # Async cleanup
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass  # Ignore cleanup errors


@router.post("/process-batch")
async def process_pdf_batch(files: List[UploadFile] = File(...)):
    """
    Process multiple PDF files concurrently for maximum throughput.
    
    This endpoint processes multiple PDFs in parallel, significantly faster
    than processing them one by one.
    
    Returns:
        List of extraction results with timing information
    """
    start_time = time.time()
    
    if not files:
        raise HTTPException(
            status_code=400,
            detail="No files provided"
        )
    
    # Validate all files first
    for file in files:
        if not file.filename or not file.filename.lower().endswith('.pdf'):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {file.filename}. All files must be PDFs."
            )
    
    # Limit batch size to prevent resource exhaustion
    max_batch_size = getattr(settings, 'MAX_BATCH_PDF_EXTRACTION', 10)
    if len(files) > max_batch_size:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files. Maximum {max_batch_size} files per batch."
        )
    
    temp_paths = []
    results = []
    
    try:
        # Phase 1: Async file saving (parallel I/O)
        save_start = time.time()
        save_tasks = []
        
        for file in files:
            temp_filename = f"{uuid.uuid4()}_{file.filename}"
            temp_path = os.path.join(tempfile.gettempdir(), temp_filename)
            temp_paths.append(temp_path)
            
            # Create async task for each file save
            async def save_file_task(f, path):
                content = await f.read()
                await save_file_async(content, path)
                return path
            
            save_tasks.append(save_file_task(file, temp_path))
        
        # Wait for all files to be saved
        await asyncio.gather(*save_tasks)
        save_time = time.time() - save_start
        
        # Phase 2: Parallel PDF extraction (CPU-intensive)
        extract_start = time.time()
        
        # Use ThreadPoolExecutor for CPU-bound PDF processing
        max_workers = min(len(files), settings.MAX_CONCURRENT_UPLOADS)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all extraction tasks
            extraction_futures = [
                asyncio.get_event_loop().run_in_executor(
                    executor, extract_pdf_sync, temp_path
                ) for temp_path in temp_paths
            ]
            
            # Wait for all extractions to complete
            results = await asyncio.gather(*extraction_futures, return_exceptions=True)
        
        extract_time = time.time() - extract_start
        total_time = time.time() - start_time
        
        # Process results and handle errors
        processed_results = []
        successful_count = 0
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "filename": files[i].filename,
                    "success": False,
                    "error": str(result),
                    "title": None,
                    "outline": [],
                    "page_count": 0
                })
            else:
                processed_results.append({
                    "filename": files[i].filename,
                    "success": result.success,
                    "title": result.title,
                    "outline": [
                        {
                            "level": item.level,
                            "title": item.title,
                            "page_number": item.page_number + 1  # Convert to 1-indexed for display
                        } for item in result.outline
                    ],
                    "page_count": result.page_count,
                    "processing_time": result.processing_time
                })
                if result.success:
                    successful_count += 1
        
        # Log performance metrics
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"🚀 Batch PDF extraction: {total_time:.2f}s total for {len(files)} files "
                   f"(save: {save_time:.2f}s, extract: {extract_time:.2f}s, "
                   f"successful: {successful_count}/{len(files)})")
        
        return {
            "results": processed_results,
            "summary": {
                "total_files": len(files),
                "successful": successful_count,
                "failed": len(files) - successful_count,
                "total_time": total_time,
                "save_time": save_time,
                "extract_time": extract_time,
                "files_per_second": len(files) / total_time if total_time > 0 else 0
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch PDF processing failed: {str(e)}"
        )
    
    finally:
        # Cleanup all temporary files
        for temp_path in temp_paths:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass


@router.post("/process-pdf-legacy")
async def process_pdf_legacy(file: UploadFile = File(...)):
    """
    Legacy endpoint for backwards compatibility with original Challenge 1a API.
    Returns the same data structure as the original API.
    """
    # Process using the main endpoint
    result = await process_pdf(file)
    
    # Convert to legacy format
    legacy_response = {
        "title": result.title or "",
        "outline": [
            {
                "text": item.title,
                "level": item.level,
                "page": item.page_number + 1  # Convert to 1-indexed for display
            }
            for item in result.outline
        ]
    }
    
    # Add error field if processing failed
    if not result.success:
        legacy_response["error"] = result.message or "Processing failed"
    
    return legacy_response


@router.get("/health")
async def extraction_health():
    """Health check for PDF extraction service."""
    try:
        # Test if extraction service is available
        from ..services.pdf_extract import pdf_extract_service
        
        capabilities = {
            "pdf_extraction": True,
            "gemini_enhancement": bool(pdf_extract_service.gemini_available),
            "supported_formats": ["pdf"]
        }
        
        return {
            "status": "healthy",
            "service": "pdf_extraction",
            "capabilities": capabilities,
            "timestamp": time.time()
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time()
            }
        )
