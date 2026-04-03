from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from jinja2 import Environment, FileSystemLoader, select_autoescape
from pathlib import Path
import threading
import db
import indexer

app = FastAPI(title="FastPhoto")

env = Environment(
    loader=FileSystemLoader("templates"),
    autoescape=select_autoescape(),
    cache_size=0,
)
templates = Jinja2Templates(env=env)

# Background indexing lock
indexing_lock = threading.Lock()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main search page."""
    return templates.TemplateResponse(request, "index.html")


@app.get("/search", response_class=HTMLResponse)
async def search(request: Request, q: str = ""):
    """Search photos and return HTML fragment for HTMX."""
    if not q.strip():
        return templates.TemplateResponse(request, "results.html", {
            "results": [],
            "query": "",
            "count": 0,
        })

    results = indexer.search(q, db_path=db.DB_PATH)
    if not results:
        results = db.search_legacy(q, db_path=db.DB_PATH)

    return templates.TemplateResponse(request, "results.html", {
        "results": results,
        "query": q,
        "count": len(results),
    })


@app.get("/photos/{filepath:path}")
async def serve_photo(filepath: str):
    """Serve the actual image file."""
    file_path = Path(filepath)
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Image not found")

    try:
        file_path.resolve().relative_to(Path.cwd().resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")

    return FileResponse(file_path)


@app.get("/api/settings")
async def get_settings():
    """Return available providers and models."""
    return JSONResponse(indexer.PROVIDERS)


@app.get("/api/index/status")
async def get_index_status():
    """Return current indexing progress."""
    return JSONResponse(indexer.progress.get())


@app.get("/api/stats")
async def get_stats():
    """Return database statistics."""
    try:
        return JSONResponse({
            "total": db.get_photo_count(),
            "errors": db.get_error_count(),
        })
    except Exception:
        return JSONResponse({"total": 0, "errors": 0})


@app.post("/api/index")
async def start_index(
    folder: str = "./photos",
    recursive: bool = True,
    provider: str = "anthropic",
    model: str = "",
    delay: float = 0.5,
):
    """Start indexing photos in the background."""
    if indexing_lock.locked():
        return JSONResponse({"error": "Indexing already in progress"}, status_code=409)

    if not Path(folder).exists():
        return JSONResponse({"error": f"Folder not found: {folder}"}, status_code=400)

    if not model:
        model = indexer.PROVIDERS.get(provider, {}).get("default_model")

    def run_index():
        with indexing_lock:
            try:
                indexer.process_folder(
                    folder=folder,
                    db_path=db.DB_PATH,
                    rate_limit_delay=delay,
                    recursive=recursive,
                    provider=provider,
                    model=model,
                )
            except Exception as e:
                indexer.progress.update("", 0, 0, f"Error: {e}")
                indexer.progress.finish()

    threading.Thread(target=run_index, daemon=True).start()
    return JSONResponse({"message": "Indexing started"})


@app.delete("/api/photos/{filepath:path}")
async def delete_photo(filepath: str):
    """Delete a photo from the database."""
    if db.delete_photo(filepath, db_path=db.DB_PATH):
        return JSONResponse({"message": f"Deleted: {filepath}"})
    return JSONResponse({"error": "Photo not found"}, status_code=404)


@app.post("/api/rebuild-fts")
async def rebuild_fts():
    """Rebuild the FTS search index."""
    if indexing_lock.locked():
        return JSONResponse({"error": "Cannot rebuild while indexing is in progress"}, status_code=409)
    db.rebuild_fts()
    return JSONResponse({"message": "FTS index rebuilt"})
