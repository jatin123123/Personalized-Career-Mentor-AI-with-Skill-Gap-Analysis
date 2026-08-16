from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.analysis import router as analysis_router
from routes.extension import router as extension_router

app = FastAPI(title="AI Resume Matcher API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Chrome extension has chrome-extension:// origin
    allow_credentials=False,      # must be False when allow_origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analysis_router, prefix="/api")
app.include_router(extension_router, prefix="/api")


@app.get("/")
def health():
    return {"status": "ok", "message": "AI Resume Matcher API is running"}
