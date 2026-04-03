import sqlite3
import json
from pathlib import Path
from contextlib import contextmanager

DB_PATH = "photo_index.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS photos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filepath TEXT UNIQUE NOT NULL,
    description TEXT,
    location TEXT,
    setting TEXT,
    mood TEXT,
    time_of_day TEXT,
    season TEXT,
    provider TEXT,
    model TEXT,
    error TEXT,
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS tags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    photo_id INTEGER NOT NULL,
    tag TEXT NOT NULL,
    FOREIGN KEY (photo_id) REFERENCES photos(id) ON DELETE CASCADE
);

CREATE VIRTUAL TABLE IF NOT EXISTS photos_fts USING fts5(
    description,
    location,
    setting,
    mood,
    time_of_day,
    season,
    filepath,
    tags,
    content=''
);

CREATE INDEX IF NOT EXISTS idx_tags_photo_id ON tags(photo_id);
CREATE INDEX IF NOT EXISTS idx_tags_tag ON tags(tag);
"""


@contextmanager
def get_connection(db_path=DB_PATH):
    """Get a database connection with foreign keys enabled."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path=DB_PATH):
    """Initialize the database schema."""
    with get_connection(db_path) as conn:
        conn.executescript(SCHEMA)


def migrate_from_json(json_path="photo_index.json", db_path=DB_PATH):
    """Migrate existing JSON index to SQLite."""
    json_file = Path(json_path)
    if not json_file.exists():
        print(f"No JSON file found at {json_path}, skipping migration")
        return

    print(f"Migrating from {json_path}...")
    with open(json_file) as f:
        data = json.load(f)

    init_db(db_path)

    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        migrated = 0
        skipped = 0

        for entry in data:
            filepath = entry.get("filepath", "")
            if not filepath:
                continue

            # Check if already exists
            cursor.execute("SELECT id FROM photos WHERE filepath = ?", (filepath,))
            if cursor.fetchone():
                skipped += 1
                continue

            error = entry.get("error")
            if error:
                cursor.execute(
                    "INSERT OR IGNORE INTO photos (filepath, error) VALUES (?, ?)",
                    (filepath, error),
                )
                continue

            cursor.execute(
                """INSERT OR IGNORE INTO photos
                   (filepath, description, location, setting, mood, time_of_day, season, provider, model)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    filepath,
                    entry.get("description"),
                    entry.get("location"),
                    entry.get("setting"),
                    entry.get("mood"),
                    entry.get("time_of_day"),
                    entry.get("season"),
                    entry.get("provider", "unknown"),
                    entry.get("model", "unknown"),
                ),
            )

            photo_id = cursor.lastrowid
            if photo_id == 0:
                # Photo already existed, get its ID
                cursor.execute("SELECT id FROM photos WHERE filepath = ?", (filepath,))
                row = cursor.fetchone()
                if row:
                    photo_id = row["id"]

            # Insert tags (triggers FTS update)
            for tag in entry.get("tags", []):
                cursor.execute(
                    "INSERT INTO tags (photo_id, tag) VALUES (?, ?)",
                    (photo_id, tag),
                )

            migrated += 1

        print(f"Migration complete: {migrated} new, {skipped} skipped")


def insert_photo(filepath, description=None, location=None, setting=None,
               mood=None, time_of_day=None, season=None, provider=None,
               model=None, error=None, tags=None, db_path=DB_PATH):
    """Insert a photo record into the database and update FTS. Returns photo_id."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()

        if error:
            cursor.execute(
                "INSERT OR IGNORE INTO photos (filepath, error) VALUES (?, ?)",
                (filepath, error),
            )
            return None

        cursor.execute(
            """INSERT OR IGNORE INTO photos
               (filepath, description, location, setting, mood, time_of_day, season, provider, model)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (filepath, description, location, setting, mood, time_of_day, season, provider, model),
        )

        photo_id = cursor.lastrowid
        if photo_id == 0:
            cursor.execute("SELECT id FROM photos WHERE filepath = ?", (filepath,))
            row = cursor.fetchone()
            if row:
                photo_id = row["id"]

        # Insert tags
        if tags:
            for tag in tags:
                cursor.execute(
                    "INSERT INTO tags (photo_id, tag) VALUES (?, ?)",
                    (photo_id, tag),
                )

        # Update FTS
        _update_fts(conn, photo_id, description, location, setting, mood, time_of_day, season, filepath, tags)

        return photo_id


def _update_fts(conn, photo_id, description, location, setting, mood, time_of_day, season, filepath, tags):
    """Update FTS5 index for a photo."""
    tags_text = " ".join(tags) if tags else ""
    conn.execute(
        """INSERT OR REPLACE INTO photos_fts(rowid, description, location, setting, mood, time_of_day, season, filepath, tags)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (photo_id, description, location, setting, mood, time_of_day, season, filepath, tags_text),
    )


def delete_photo(filepath, db_path=DB_PATH):
    """Delete a photo and rebuild FTS without it."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM photos WHERE filepath = ?", (filepath,))
        row = cursor.fetchone()
        if not row:
            return False

        photo_id = row["id"]

        # Delete tags and photo
        conn.execute("DELETE FROM tags WHERE photo_id = ?", (photo_id,))
        conn.execute("DELETE FROM photos WHERE id = ?", (photo_id,))

        # Rebuild FTS (content='' doesn't support individual deletes)
        _rebuild_fts_internal(conn)

        return True


def _rebuild_fts_internal(conn):
    """Internal FTS rebuild using existing connection."""
    conn.execute("DROP TABLE IF EXISTS photos_fts")
    conn.execute("""CREATE VIRTUAL TABLE photos_fts USING fts5(
        description, location, setting, mood, time_of_day, season, filepath, tags, content=''
    )""")

    cursor = conn.cursor()
    cursor.execute("SELECT * FROM photos")
    for row in cursor.fetchall():
        photo = dict(row)
        cursor.execute("SELECT tag FROM tags WHERE photo_id = ?", (photo["id"],))
        tags = [r["tag"] for r in cursor.fetchall()]
        tags_text = " ".join(tags)
        conn.execute(
            """INSERT INTO photos_fts(rowid, description, location, setting, mood, time_of_day, season, filepath, tags)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (photo["id"], photo["description"], photo["location"], photo["setting"],
             photo["mood"], photo["time_of_day"], photo["season"], photo["filepath"], tags_text),
        )


def get_processed_filepaths(db_path=DB_PATH):
    """Get set of already processed filepaths."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT filepath FROM photos")
        return {row["filepath"] for row in cursor.fetchall()}


def search(query, limit=30, offset=0, db_path=DB_PATH):
    """Search photos using FTS5 full-text search."""
    if not query:
        return []

    with get_connection(db_path) as conn:
        cursor = conn.cursor()

        # Use FTS5 MATCH for ranked results
        fts_query = " OR ".join(query.lower().split())
        cursor.execute(
            """SELECT p.*, rank
               FROM photos_fts
               JOIN photos p ON photos_fts.rowid = p.id
               WHERE photos_fts MATCH ?
               ORDER BY rank
               LIMIT ? OFFSET ?""",
            (fts_query, limit, offset),
        )

        results = []
        for row in cursor.fetchall():
            photo = dict(row)
            # Fetch tags for this photo
            cursor.execute("SELECT tag FROM tags WHERE photo_id = ?", (photo["id"],))
            photo["tags"] = [r["tag"] for r in cursor.fetchall()]
            results.append(photo)

        return results


def search_legacy(query, limit=30, offset=0, db_path=DB_PATH):
    """Fallback keyword search without FTS5 (for compatibility)."""
    if not query:
        return []

    query_terms = query.lower().split()

    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM photos WHERE error IS NULL")

        scored = []
        for row in cursor.fetchall():
            photo = dict(row)

            # Fetch tags
            cursor.execute("SELECT tag FROM tags WHERE photo_id = ?", (photo["id"],))
            tags = [r["tag"] for r in cursor.fetchall()]
            photo["tags"] = tags

            # Build searchable text
            searchable = " ".join([
                photo.get("description", "") or "",
                photo.get("location", "") or "",
                photo.get("setting", "") or "",
                photo.get("mood", "") or "",
                photo.get("time_of_day", "") or "",
                photo.get("season", "") or "",
                " ".join(tags),
                photo.get("filepath", ""),
            ]).lower()

            score = sum(term in searchable for term in query_terms)
            if score > 0:
                scored.append((score, photo))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [photo for _, photo in scored[offset:offset + limit]]


def search_count(query, db_path=DB_PATH):
    """Get total count of matching photos using FTS5."""
    if not query:
        return 0

    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        fts_query = " OR ".join(query.lower().split())
        cursor.execute(
            """SELECT COUNT(*) as count
               FROM photos_fts
               WHERE photos_fts MATCH ?""",
            (fts_query,),
        )
        row = cursor.fetchone()
        return row["count"] if row else 0


def get_photo_count(db_path=DB_PATH):
    """Get total number of processed photos."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as count FROM photos")
        return cursor.fetchone()["count"]


def get_error_count(db_path=DB_PATH):
    """Get number of photos that failed processing."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as count FROM photos WHERE error IS NOT NULL")
        return cursor.fetchone()["count"]


def export_to_json(json_path="photo_index.json", db_path=DB_PATH):
    """Export database contents back to JSON format."""
    with get_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM photos")

        results = []
        for row in cursor.fetchall():
            photo = dict(row)
            # Fetch tags for this photo
            cursor.execute("SELECT tag FROM tags WHERE photo_id = ?", (photo["id"],))
            photo["tags"] = [r["tag"] for r in cursor.fetchall()]
            results.append(photo)

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Exported {len(results)} records to {json_path}")


def rebuild_fts(db_path=DB_PATH):
    """Rebuild the FTS index from scratch using current photos and tags data."""
    with get_connection(db_path) as conn:
        _rebuild_fts_internal(conn)
        print("FTS index rebuilt")
