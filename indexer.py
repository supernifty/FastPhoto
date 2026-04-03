import anthropic
import base64
import json
import time
import argparse
import threading
from pathlib import Path
from dotenv import load_dotenv
import os
from PIL import Image
import io
from openai import OpenAI
import dashscope
from dashscope import MultiModalConversation
import db

load_dotenv()

# Initialize clients for all providers
anthropic_client = anthropic.Anthropic()
openai_client = OpenAI()
dashscope.api_key = os.getenv("DASHSCOPE_API_KEY", "")

# Claude API limit: 5MB (base64 encoded)
# Base64 increases size by ~33%, so target ~3MB raw to stay safely under limit
MAX_IMAGE_BYTES = 3 * 1024 * 1024  # 3MB raw → ~4MB base64

# Progress tracking (shared state for background indexing)
class IndexProgress:
    def __init__(self):
        self._lock = threading.Lock()
        self._reset()

    def _reset(self):
        self.active = False
        self.total = 0
        self.processed = 0
        self.errors = 0
        self.current_file = ""
        self.message = ""

    def start(self, total: int):
        with self._lock:
            self._reset()
            self.active = True
            self.total = total

    def update(self, current_file: str, processed: int, errors: int, message: str = ""):
        with self._lock:
            self.current_file = current_file
            self.processed = processed
            self.errors = errors
            self.message = message

    def finish(self):
        with self._lock:
            self.active = False

    def get(self) -> dict:
        with self._lock:
            return {
                "active": self.active,
                "total": self.total,
                "processed": self.processed,
                "errors": self.errors,
                "current_file": self.current_file,
                "message": self.message,
            }

progress = IndexProgress()


def downsample_image(image_path: Path, max_size_bytes: int = MAX_IMAGE_BYTES) -> tuple[bytes, str]:
    """Downsample image if it exceeds max size. Returns (image_data, media_type)."""

    suffix_to_type = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp"
    }
    media_type = suffix_to_type.get(image_path.suffix.lower(), "image/jpeg")

    with open(image_path, "rb") as f:
        original_data = f.read()

    if len(original_data) <= max_size_bytes:
        b64_size = len(base64.standard_b64encode(original_data))
        if b64_size <= MAX_IMAGE_BYTES:
            return original_data, media_type
        print(f"  Note: Base64 size ({b64_size // 1024}KB) exceeds limit, downsampling...")

    print(f"  Downsampling ({len(original_data) // 1024}KB raw -> target <{max_size_bytes // 1024}KB)...")

    with Image.open(image_path) as img:
        if img.mode in ('RGBA', 'LA', 'P'):
            img = img.convert('RGB')

        width, height = img.size
        scale = 1.0

        for quality in [85, 75, 65, 55, 45]:
            if scale < 1.0:
                new_size = (int(width * scale), int(height * scale))
                img_resized = img.resize(new_size, Image.Resampling.LANCZOS)
            else:
                img_resized = img

            buffer = io.BytesIO()
            img_resized.save(buffer, format="JPEG", quality=quality, optimize=True)
            data = buffer.getvalue()

            if len(data) <= max_size_bytes:
                b64_check = len(base64.standard_b64encode(data))
                if b64_check <= MAX_IMAGE_BYTES:
                    return data, "image/jpeg"

            scale *= 0.75

        final_size = (int(width * 0.5), int(height * 0.5))
        img_final = img.resize(final_size, Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        img_final.save(buffer, format="JPEG", quality=40, optimize=True)
        data = buffer.getvalue()
        b64_final = len(base64.standard_b64encode(data))
        print(f"  Warning: Had to downscale significantly (final: {b64_final // 1024}KB base64)")
        return data, "image/jpeg"


IMAGE_DESCRIPTION_PROMPT = """Describe this photo for a searchable personal photo database.
Return JSON only with these fields:
{
  "description": "one or two sentence natural description",
  "people": ["brief descriptions of people visible and what they are doing, e.g. 'man grilling at BBQ', 'child blowing out birthday candles'"],
  "location": "location if identifiable, otherwise null",
  "setting": "indoor/outdoor/unknown",
  "tags": ["10-15 relevant search tags"],
  "mood": "mood or atmosphere of the photo",
  "time_of_day": "morning/afternoon/evening/night/unknown",
  "season": "spring/summer/autumn/winter/unknown"
}"""


def describe_image_anthropic(image_path: Path, model: str = "claude-haiku-4-5-20251001") -> dict:
    """Get a searchable description of an image using Anthropic Claude."""

    image_bytes, media_type = downsample_image(image_path)
    image_data = base64.standard_b64encode(image_bytes).decode("utf-8")

    response = anthropic_client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_data,
                        },
                    },
                    {"type": "text", "text": IMAGE_DESCRIPTION_PROMPT},
                ],
            }
        ],
    )

    text = response.content[0].text.strip()
    if text.startswith("```"):
        text = "\n".join(text.split("\n")[1:-1])

    result = json.loads(text)
    result["filename"] = image_path.name
    result["filepath"] = str(image_path)
    result["provider"] = "anthropic"
    result["model"] = model
    return result


def describe_image_openai(image_path: Path, model: str = "gpt-4o-mini") -> dict:
    """Get a searchable description of an image using OpenAI GPT-4o."""

    image_bytes, media_type = downsample_image(image_path)
    image_data = base64.standard_b64encode(image_bytes).decode("utf-8")

    response = openai_client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{image_data}"
                        },
                    },
                    {"type": "text", "text": IMAGE_DESCRIPTION_PROMPT},
                ],
            }
        ],
        max_tokens=1024,
    )

    text = response.choices[0].message.content.strip()
    if text.startswith("```"):
        text = "\n".join(text.split("\n")[1:-1])

    result = json.loads(text)
    result["filename"] = image_path.name
    result["filepath"] = str(image_path)
    result["provider"] = "openai"
    result["model"] = model
    return result


def describe_image_qwen(image_path: Path, model: str = "qwen3-vl-flash") -> dict:
    """Get a searchable description of an image using Qwen VL via DashScope."""

    image_bytes, media_type = downsample_image(image_path)
    image_data = base64.standard_b64encode(image_bytes).decode("utf-8")

    messages = [
        {
            "role": "user",
            "content": [
                {"image": f"data:{media_type};base64,{image_data}"},
                {"text": IMAGE_DESCRIPTION_PROMPT},
            ],
        }
    ]

    response = MultiModalConversation.call(
        model=model,
        messages=messages,
    )

    if response.status_code != 200:
        raise Exception(f"DashScope error: {response.code} - {response.message}")

    text = response.output.choices[0].message.content[0]["text"].strip()
    if text.startswith("```"):
        text = "\n".join(text.split("\n")[1:-1])

    result = json.loads(text)
    result["filename"] = image_path.name
    result["filepath"] = str(image_path)
    result["provider"] = "qwen"
    result["model"] = model
    return result


def describe_image(image_path: str, provider: str = "anthropic", model: str = None) -> dict:
    """Get a searchable description of an image using specified provider."""

    image_path = Path(image_path)

    if provider == "openai":
        model = model or "gpt-4o-mini"
        return describe_image_openai(image_path, model)
    elif provider == "qwen":
        model = model or "qwen3-vl-flash"
        return describe_image_qwen(image_path, model)
    else:
        model = model or "claude-haiku-4-5-20251001"
        return describe_image_anthropic(image_path, model)


PROVIDERS = {
    "anthropic": {
        "default_model": "claude-haiku-4-5-20251001",
        "models": ["claude-haiku-4-5-20251001", "claude-sonnet-4-5-20250929", "claude-opus-4-5-20251101"],
    },
    "openai": {
        "default_model": "gpt-4o-mini",
        "models": ["gpt-4o-mini", "gpt-4o"],
    },
    "qwen": {
        "default_model": "qwen3-vl-flash",
        "models": ["qwen3-vl-flash", "qwen3-vl-plus", "qwen-vl-max"],
    },
}


def process_folder(
    folder: str,
    db_path: str = db.DB_PATH,
    rate_limit_delay: float = 0.5,
    recursive: bool = True,
    provider: str = "anthropic",
    model: str = None
):
    """Process all images in a folder and save descriptions."""

    folder = Path(folder)
    extensions = {".jpg", ".jpeg", ".png", ".gif", ".webp"}

    db.init_db(db_path)

    glob_func = folder.rglob if recursive else folder.glob
    image_files = [f for f in glob_func("*") if f.suffix.lower() in extensions]

    processed = db.get_processed_filepaths(db_path)
    remaining = [f for f in image_files if str(f) not in processed]

    print(f"Found {len(image_files)} images, {len(remaining)} unprocessed")
    print(f"Provider: {provider}, Model: {model or 'default'}")

    progress.start(len(remaining))

    for i, image_path in enumerate(remaining):
        print(f"Processing {i+1}/{len(remaining)}: {image_path.name}")
        progress.update(image_path.name, i, progress.errors, f"Processing {i+1}/{len(remaining)}")

        try:
            desc = describe_image(image_path, provider=provider, model=model)

            db.insert_photo(
                filepath=str(image_path),
                description=desc.get("description"),
                location=desc.get("location"),
                setting=desc.get("setting"),
                mood=desc.get("mood"),
                time_of_day=desc.get("time_of_day"),
                season=desc.get("season"),
                provider=desc.get("provider"),
                model=desc.get("model"),
                tags=desc.get("tags", []),
                db_path=db_path,
            )

            print(f"  → {desc.get('description', '')[:80]}...")
            time.sleep(rate_limit_delay)

        except Exception as e:
            print(f"  Failed: {e}")
            db.insert_photo(filepath=str(image_path), error=str(e), db_path=db_path)
            progress.update(image_path.name, i + 1, progress.errors + 1, f"Failed: {e}")

    count = db.get_photo_count(db_path)
    errors = db.get_error_count(db_path)
    msg = f"Done. {count} photos in database ({errors} with errors)"
    print(f"\n{msg}")
    progress.update("", len(remaining), progress.errors, msg)
    progress.finish()


def search(query: str, db_path: str = db.DB_PATH) -> list:
    """Search photos using FTS5 full-text search."""
    results = db.search(query, db_path=db_path)
    if not results:
        results = db.search_legacy(query, db_path=db_path)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI-powered photo metadata generator")
    parser.add_argument("folder", nargs="?", default="./photos", help="Folder to process (default: ./photos)")
    parser.add_argument("--db", default=db.DB_PATH, help=f"Database file (default: {db.DB_PATH})")
    parser.add_argument("--no-recursive", action="store_true", help="Disable recursive processing of subdirectories")
    parser.add_argument("--delay", type=float, default=0.5, help="Rate limit delay in seconds (default: 0.5)")
    parser.add_argument("-p", "--provider", choices=["anthropic", "openai", "qwen"], default="anthropic", help="AI provider (default: anthropic)")
    parser.add_argument("-m", "--model", help="Model name (default: claude-haiku-4-5-20251001 for Anthropic, gpt-4o-mini for OpenAI)")
    parser.add_argument("--search", metavar="QUERY", help="Search the database instead of processing images")
    parser.add_argument("--migrate", action="store_true", help="Migrate existing JSON index to SQLite")
    parser.add_argument("--export", action="store_true", help="Export database to JSON file")
    parser.add_argument("--delete", metavar="FILEPATH", help="Delete a photo from the database by filepath")
    parser.add_argument("--rebuild-fts", action="store_true", help="Rebuild the FTS search index")

    args = parser.parse_args()

    if args.delete:
        if db.delete_photo(args.delete, db_path=args.db):
            print(f"Deleted: {args.delete}")
        else:
            print(f"Not found: {args.delete}")
    elif args.rebuild_fts:
        db.rebuild_fts(db_path=args.db)
    elif args.export:
        db.export_to_json(db_path=args.db)
    elif args.migrate:
        db.init_db(args.db)
        db.migrate_from_json(db_path=args.db)
    elif args.search:
        results = search(args.search, db_path=args.db)
        for r in results[:30]:
            print(r["filepath"])
            print(r.get("description", ""))
            print()
    else:
        process_folder(
            folder=args.folder,
            db_path=args.db,
            rate_limit_delay=args.delay,
            recursive=not args.no_recursive,
            provider=args.provider,
            model=args.model
        )
