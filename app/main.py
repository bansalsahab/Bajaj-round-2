from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from .models import LabTestResponse, LabTest
from .processing import process_lab_report

app = FastAPI(
    title="Lab Report Extractor",
    description="Extracts lab test data from scanned medical lab reports using computer vision and OCR",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/get-lab-tests", response_model=LabTestResponse)
async def get_lab_tests(file: UploadFile = File(...)):
    """
    Process a lab report image and extract test information.
    
    Args:
        file: The uploaded lab report image (PNG/JPEG)
        
    Returns:
        JSON with extracted lab test data
    """
    try:
        # Check file type
        if file.content_type not in ["image/png", "image/jpeg"]:
            raise HTTPException(status_code=400, detail="Only PNG and JPEG images are supported")
        
        # Read file content
        image_data = await file.read()
        
        # Process the image and get lab tests
        lab_tests = process_lab_report(image_data)
        
        # Create response
        response = LabTestResponse(
            is_success=True,
            data=lab_tests
        )
        return response
    
    except Exception as e:
        # Log the error (in a production system)
        print(f"Error processing lab report: {str(e)}")
        
        # Return failure response
        return LabTestResponse(
            is_success=False,
            data=[]
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
