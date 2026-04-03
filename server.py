from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from jinja2 import Environment, FileSystemLoader, select_autoescape
from pathlib import Path
import threading
import os
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

# Whether to restrict directory browser to home directory
RESTRICT_TO_HOME = os.getenv("FASTPHOTO_RESTRICT_HOME", "true").lower() == "true"

# Whether to restrict photo serving to current working directory
RESTRICT_TO_CWD = os.getenv("FASTPHOTO_RESTRICT_CWD", "true").lower() == "true"


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main search page."""
    return templates.TemplateResponse(request, "index.html")


@app.get("/search", response_class=HTMLResponse)
async def search(request: Request, q: str = "", offset: int = 0, limit: int = 30):
    """Search photos and return HTML fragment for HTMX."""
    if not q.strip():
        return templates.TemplateResponse(request, "results.html", {
            "results": [],
            "query": "",
            "total": 0,
            "offset": 0,
            "limit": limit,
            "has_more": False,
        })

    results = indexer.search(q, db_path=db.DB_PATH, limit=limit, offset=offset)
    if not results:
        results = db.search_legacy(q, db_path=db.DB_PATH, limit=limit, offset=offset)

    # Get total count
    total = db.search_count(q, db_path=db.DB_PATH)
    if total == 0 and results:
        # Fallback count from legacy search
        total = len(results) + offset

    has_more = (offset + len(results)) < total

    return templates.TemplateResponse(request, "results.html", {
        "results": results,
        "query": q,
        "total": total,
        "offset": offset,
        "limit": limit,
        "has_more": has_more,
    })


@app.get("/photo/{photo_id}")
async def photo_detail(request: Request, photo_id: int):
    """Show full details for a single photo."""
    import sqlite3
    conn = sqlite3.connect(db.DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM photos WHERE id = ?", (photo_id,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Photo not found")

    photo = dict(row)
    cursor.execute("SELECT tag FROM tags WHERE photo_id = ?", (photo_id,))
    photo["tags"] = [r["tag"] for r in cursor.fetchall()]
    conn.close()

    return templates.TemplateResponse(request, "photo_detail.html", {"photo": photo})


@app.get("/thumbnails/{photo_id}")
async def serve_thumbnail(photo_id: int):
    """Serve a thumbnail image by photo ID."""
    thumb_path = Path("thumbnails") / f"{photo_id}.jpg"
    if not thumb_path.exists() or not thumb_path.is_file():
        # Fall back to full image — look up filepath from DB
        import sqlite3
        conn = sqlite3.connect(db.DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT filepath FROM photos WHERE id = ?", (photo_id,))
        row = cursor.fetchone()
        conn.close()
        if row:
            return await serve_photo(row["filepath"])
        raise HTTPException(status_code=404, detail="Thumbnail not found")

    return FileResponse(thumb_path, media_type="image/jpeg")


@app.get("/photos/{filepath:path}")
async def serve_photo(filepath: str):
    """Serve the actual image file."""
    file_path = Path(filepath)
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Image not found")

    if RESTRICT_TO_CWD:
        try:
            file_path.resolve().relative_to(Path.cwd().resolve())
        except ValueError:
            raise HTTPException(status_code=403, detail="Access denied. Set FASTPHOTO_RESTRICT_CWD=false to allow serving files outside the current directory.")

    return FileResponse(file_path)


@app.get("/api/directories")
async def list_directories(path: str = "."):
    """List subdirectories at a given path for the folder browser."""
    try:
        base = Path(path).resolve()

        if RESTRICT_TO_HOME:
            home = Path.home().resolve()
            if not str(base).startswith(str(home)):
                return JSONResponse({"error": "Access denied"}, status_code=403)

        if not base.exists() or not base.is_dir():
            return JSONResponse({"error": "Path not found"}, status_code=404)

        entries = []
        for item in sorted(base.iterdir()):
            if item.is_dir() and not item.name.startswith("."):
                entries.append({
                    "name": item.name,
                    "path": str(item),
                    "parent": str(base.parent),
                })

        return JSONResponse({
            "current": str(base),
            "parent": str(base.parent) if (not RESTRICT_TO_HOME or base != home) else None,
            "directories": entries,
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


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


@app.post("/api/rebuild-thumbnails")
async def rebuild_thumbnails_endpoint():
    """Rebuild thumbnails in the background."""
    if indexing_lock.locked():
        return JSONResponse({"error": "Indexing already in progress"}, status_code=409)

    def run_rebuild():
        with indexing_lock:
            try:
                indexer.rebuild_thumbnails()
            except Exception as e:
                indexer.progress.update("", 0, 0, f"Error: {e}")
                indexer.progress.finish()

    threading.Thread(target=run_rebuild, daemon=True).start()
    return JSONResponse({"message": "Thumbnail rebuild started"})
