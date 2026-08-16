from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.analysis import router as analysis_router
from routes.extension import router as extension_router

app = FastAPI(title="AI Resume Matcher API", version="1.0.0")

# Allowed origins: Vercel frontend domains + Chrome extension origins
ALLOWED_ORIGINS = [
    # Vercel production + preview domains
    "https://career-mentor-ai.vercel.app",
    "https://*.vercel.app",
    # Local development
    "http://localhost:3000",
    "http://localhost:3001",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=r"https://.*\.vercel\.app|chrome-extension://.*",
    allow_credentials=False,      # must be False when using regex/wildcard origins
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Accept", "Authorization"],
)

app.include_router(analysis_router, prefix="/api")
app.include_router(extension_router, prefix="/api")


@app.get("/")
def health():
    return {"status": "ok", "message": "AI Resume Matcher API is running"}
