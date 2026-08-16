from pydantic import BaseModel, Field
from typing import Optional, List, Union


class AnalyzeRequest(BaseModel):
    job_description: str
    resume_text: str
    model: str = "llama-3.3-70b-versatile"
    depth: str = "Standard"
    temperature: float = 0.1


class AnalysisResult(BaseModel):
    skills_matched: List[str] = Field(default_factory=list, description="Skills present in both JD and Resume")
    skills_missing: List[str] = Field(default_factory=list, description="Explicit JD requirements/skills NOT present in resume")
    skills_extra: List[str] = Field(default_factory=list, description="Candidate skills not required by JD but relevant/valuable")
    experience_match: str = Field(default="", description="Evaluation of experience level and industry alignment")
    education_match: str = Field(default="", description="Evaluation of education credentials vs JD requirements")
    overall_match_percentage: int = Field(default=0, description="Overall match percentage 0-100")
    selection_probability: str = Field(default="Medium", description="High / Medium / Low selection probability")
    strength_areas: List[str] = Field(default_factory=list, description="Key candidate strengths aligned with this specific job")
    improvement_areas: List[str] = Field(default_factory=list, description="Specific gap areas to address for this role")
    specific_recommendations: List[str] = Field(default_factory=list, description="Actionable bullet points to improve fit for THIS specific role")
    interview_preparation: List[str] = Field(default_factory=list, description="Targeted technical and domain interview questions for this specific position")
    salary_competitiveness: str = Field(default="", description="Salary competitiveness or position benchmark")


class AnalyzeResponse(BaseModel):
    success: bool
    result: Optional[AnalysisResult] = None
    error: Optional[str] = None


class UploadResumeResponse(BaseModel):
    success: bool
    text: Optional[str] = None
    char_count: Optional[int] = None
    error: Optional[str] = None
