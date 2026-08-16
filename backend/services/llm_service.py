import os
import json
from pathlib import Path
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from models.schemas import AnalysisResult

# On Azure/cloud, GROQ_API_KEY is set as an App Setting (environment variable).
# Locally, it's read from backend/.env via python-dotenv (loaded in __init__ or startup).
# We deliberately do NOT call load_dotenv() here so cloud env vars take precedence cleanly.
try:
    from dotenv import load_dotenv
    # Only load .env files in local development (when not running on Azure)
    if not os.getenv("WEBSITE_SITE_NAME"):  # WEBSITE_SITE_NAME is always set on Azure App Service
        _BASE_DIR = Path(__file__).resolve().parents[2]
        load_dotenv(_BASE_DIR / ".env")
        load_dotenv(_BASE_DIR / "backend" / ".env", override=True)
except ImportError:
    pass

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is not set. Set it as an Azure App Setting or in backend/.env for local dev.")

_parser = JsonOutputParser(pydantic_object=AnalysisResult)
_format_instructions = _parser.get_format_instructions()

_SYSTEM_MSG = """You are an expert Principal AI Career Mentor & Technical Recruiter.
Your job is to perform an ultra-accurate, domain-faithful skill-gap analysis comparing a candidate's resume against a specific target Job Description (JD).

CRITICAL RULES:
1. STRICT RELEVANCE:
   - All `skills_matched` must be skills/qualifications present in BOTH the JD and the Resume.
   - All `skills_missing` must be skills/qualifications explicitly requested or required by the TARGET JOB DESCRIPTION that are missing from the resume.
   - NEVER invent or recommend irrelevant domains. For example, if the job is for a Machine Learning Engineer / AI Engineer / Data Scientist, NEVER suggest Full-Stack, Web Development, React, Java, C++, etc., unless explicitly required in the provided JD!
   - `skills_extra` are candidate skills that are valuable or complementary to this specific domain.
2. RECOMMENDATIONS & GAPS:
   - Each recommendation in `specific_recommendations` must be a complete, actionable sentence tailored specifically to closing the real gaps for THIS specific job.
   - Do not output generic advice.
3. OUTPUT FORMAT:
   - Return strictly valid JSON following the schema. Array fields must be JSON arrays of strings (e.g. ["Python", "PyTorch", "Azure AI Foundry"])."""

_PROMPT = PromptTemplate(
    input_variables=["job_description", "resume_text", "format_instructions", "depth"],
    template="""Perform an in-depth {depth}-level resume-to-job matching analysis.

Target Job Description:
{job_description}

Candidate Resume:
{resume_text}

JSON Schema Instructions:
{format_instructions}

Remember:
- `skills_missing` must ONLY include skills/tools that are directly relevant to or required by the target job.
- `specific_recommendations` must be an array of complete, insightful recommendations addressing the actual gaps.
- Return ONLY valid JSON."""
)


def run_analysis(job_description: str, resume_text: str, model: str, depth: str, temperature: float) -> dict:
    """Call Groq LLM and parse the structured analysis result."""
    llm = ChatGroq(api_key=GROQ_API_KEY, model=model or "llama-3.3-70b-versatile", temperature=temperature)

    human_msg = HumanMessage(content=_PROMPT.format(
        job_description=job_description,
        resume_text=resume_text,
        format_instructions=_format_instructions,
        depth=depth or "Standard",
    ))

    response = llm.invoke([SystemMessage(content=_SYSTEM_MSG), human_msg])

    try:
        result = _parser.parse(response.content)
    except Exception:
        # Fallback manual JSON extract if parser encounters formatting quirks
        content = response.content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        result = json.loads(content)

    return result
