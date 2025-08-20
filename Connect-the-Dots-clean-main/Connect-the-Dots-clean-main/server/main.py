
import os
import uuid
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pdf_extractor import PDFExtractor
from fastapi.responses import FileResponse

# ...existing code...
app = FastAPI(
    title="PDF Processing API",
    description="An API to extract titles and hierarchical headings from PDF files.",
    version="1.0.0"
)
# Allow all origins for development purposes
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define a directory to store temporary files
TEMP_DIR = "temp_files"
UPLOADS_DIR = "uploads"
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)

# Upload PDF route (moved here for correct order)
@app.post("/api/upload", summary="Upload a PDF file")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF file and save it to the uploads directory.
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a PDF.")
    save_path = os.path.join(UPLOADS_DIR, file.filename)
    try:
        with open(save_path, "wb") as buffer:
            buffer.write(await file.read())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save PDF: {e}")
    return {"filename": file.filename, "url": f"/api/pdfs/{file.filename}"}
@app.get("/api/pdfs/{filename}", summary="Serve uploaded PDF files")
async def serve_pdf(filename: str):
    """
    Serve a PDF file from the uploads directory.
    """
    file_path = os.path.join(UPLOADS_DIR, filename)
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="PDF not found")
    return FileResponse(file_path, media_type="application/pdf")

@app.post("/api/process-pdf/", summary="Process a PDF to extract headings")
async def process_pdf_and_extract_headings(file: UploadFile = File(...)):
    """
    Upload a PDF file and get a structured JSON output of its headings.

    This endpoint extracts the title and a hierarchical list of headings from the
    provided PDF. Each heading includes its text, level, page number, and
    y-coordinate, which can be used to implement a "go-to" feature.
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a PDF.")

    # Create a unique temporary path for the uploaded file
    temp_pdf_path = os.path.join(TEMP_DIR, f"{uuid.uuid4()}_{file.filename}")

    import traceback
    try:
        # Save the uploaded file to the temporary path
        with open(temp_pdf_path, "wb") as buffer:
            buffer.write(await file.read())

        extractor = None
        try:
            # Process the PDF using the existing extraction logic
            extractor = PDFExtractor(temp_pdf_path)
            data = extractor.extract_data()
            if "error" in data:
                print("\n[FastAPI] Error in extract_data:", data["error"])
                return {"error": data["error"], "title": data.get("title", ""), "outline": data.get("outline", [])}
            return {
                "title": data.get("title"),
                "outline": data.get("outline", [])
            }
        finally:
            if extractor is not None:
                extractor.close()
    except Exception as e:
        print("\n[FastAPI] Exception occurred:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

    except Exception as e:
        # Handle potential errors during processing
        raise HTTPException(status_code=500, detail=f"An error occurred during PDF processing: {str(e)}")

    finally:
        # Clean up: remove the temporary file
        if os.path.exists(temp_pdf_path):
            os.remove(temp_pdf_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
