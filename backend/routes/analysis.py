from fastapi import APIRouter, UploadFile, File, HTTPException
from models.schemas import AnalyzeRequest, AnalyzeResponse, UploadResumeResponse
from services.llm_service import run_analysis
from utils.pdf_parser import extract_text_from_pdf

router = APIRouter()


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    """Run full resume-vs-JD analysis using the Groq LLM."""
    try:
        result = run_analysis(
            job_description=request.job_description,
            resume_text=request.resume_text,
            model=request.model,
            depth=request.depth,
            temperature=request.temperature,
        )
        return AnalyzeResponse(success=True, result=result)
    except Exception as e:
        return AnalyzeResponse(success=False, error=str(e))



@router.post("/upload-resume", response_model=UploadResumeResponse)
async def upload_resume(file: UploadFile = File(...)):
    """Accept a PDF upload and return extracted plain text."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    try:
        contents = await file.read()
        text = extract_text_from_pdf(contents)
        if not text:
            return UploadResumeResponse(success=False, error="Could not extract text from the PDF.")
        return UploadResumeResponse(success=True, text=text, char_count=len(text))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
