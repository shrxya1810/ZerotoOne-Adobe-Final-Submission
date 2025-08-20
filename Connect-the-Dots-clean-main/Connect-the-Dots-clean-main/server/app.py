import os
import subprocess
import uuid
import sys
import tempfile
import json
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import python_multipart
from pdf_extractor import PDFExtractor

app = FastAPI()

# Allow all origins for development purposes
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the input and output directories
INPUT_DIR = os.path.join(os.path.dirname(__file__), 'input')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')

# Ensure the directories exist
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.post("/api/process")
async def process_pdf(pdf: UploadFile = File(...)):
    """
    API endpoint to process a PDF file.
    Expects a POST request with a file part named 'pdf'.
    """
    if not pdf.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Invalid file type, please upload a PDF")

    # Generate a unique filename to avoid conflicts
    unique_filename = str(uuid.uuid4())
    input_filename = f"{unique_filename}.pdf"
    output_filename = f"{unique_filename}.json"
    input_path = os.path.join(INPUT_DIR, input_filename)
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    try:
        # Save the uploaded PDF
        with open(input_path, "wb") as buffer:
            buffer.write(await pdf.read())

        # Run the PDF processing script
        # We pass the specific file to be processed
        subprocess.run([sys.executable, 'generate_outputs.py', '--input', input_path, '--output', output_path], check=True)

        if os.path.exists(output_path):
            with open(output_path, 'r') as f:
                json_data = f.read()
            
            # Clean up the generated files
            os.remove(input_path)
            os.remove(output_path)
            
            return JSONResponse(content=json_data, media_type='application/json')
        else:
            raise HTTPException(status_code=500, detail="Failed to generate JSON output")

    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Error during PDF processing: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")


@app.post("/extract/1a/process-pdf")
async def process_pdf_extract(file: UploadFile = File(...)):
    """
    Extract title and hierarchical outline from a PDF file.
    Compatible with frontend expectations for outline/heading extraction.
    """
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
        # Save uploaded file
        content = await file.read()
        with open(temp_path, 'wb') as temp_file:
            temp_file.write(content)
        
        # Extract structure using the PDF extractor
        extractor = PDFExtractor(temp_path)
        
        # Get title and outline
        title = extractor.extract_title()
        headings_data = extractor.extract_headings()
        
        # Convert headings to expected format
        outline = []
        for heading in headings_data:
            outline_item = {
                "level": heading.get("level", 1),
                "title": heading.get("text", ""),
                "page_number": heading.get("page", 0)
            }
            outline.append(outline_item)
        
        # Clean up
        extractor.close()
        
        # Return response in expected format
        return {
            "success": True,
            "filename": file.filename,
            "title": title,
            "outline": outline,
            "page_count": len(extractor.pages_data),
            "processing_time": 0.0,
            "timestamp": None,
            "message": "PDF processed successfully"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"PDF processing failed: {str(e)}"
        )
    
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
