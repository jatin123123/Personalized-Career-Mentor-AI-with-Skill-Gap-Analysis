import io
import os
import zipfile
from pathlib import Path
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

router = APIRouter()

# Check bundled location first (for cloud deployment), then fallback to workspace root
_bundled_dir = Path(__file__).resolve().parent.parent / "extension_files"
_parent_dir = Path(__file__).resolve().parents[2] / "extension"
EXTENSION_DIR = _bundled_dir if _bundled_dir.exists() else _parent_dir

# Files to exclude from the ZIP
EXCLUDE = {".git", "__pycache__", ".DS_Store", "Thumbs.db", "README.txt"}


@router.get("/download-extension")
async def download_extension():
    """Stream the Chrome extension as a downloadable ZIP file."""
    buf = io.BytesIO()

    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(EXTENSION_DIR):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in EXCLUDE]
            for filename in files:
                if filename in EXCLUDE:
                    continue
                abs_path = Path(root) / filename
                arc_name = "resume-matcher-extension/" + abs_path.relative_to(EXTENSION_DIR).as_posix()
                zf.write(abs_path, arc_name)

    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=resume-matcher-extension.zip"},
    )
