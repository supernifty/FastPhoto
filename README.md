# FastPhoto

AI-powered photo search for your personal photo collection. Index your images with natural language descriptions, then search them instantly from a web interface or CLI.

## Features

- **AI-powered indexing** — Describes each photo with tags, location, mood, setting, and more
- **Full-text search** — Instant search across all metadata with FTS5
- **Web interface** — Clean, dark-themed photo browser with instant search
- **Multiple AI providers** — Anthropic Claude, OpenAI GPT-4o, Qwen VL
- **Background indexing** — Start indexing from the web UI with live progress
- **Large image support** — Automatic downsampling for images over 5MB
- **Resume capability** — Skips already-indexed photos on re-run

## Quick Start

```bash
# Clone and install
git clone https://github.com/yourusername/imagedb.git
cd imagedb
uv sync

# Add your API key (at least one)
cp .env.example .env
# Edit .env with your API key(s)

# Index some photos
python indexer.py ./path/to/photos

# Start the web interface
uv run uvicorn server:app --host 0.0.0.0 --port 9372
# Open http://localhost:9372
```

## Installation

### Prerequisites

- Python 3.13+
- [uv](https://github.com/astral-sh/uv) package manager
- At least one API key (see below)

### Setup

```bash
uv sync
cp .env.example .env
```

Edit `.env` and add your API key(s):

```
ANTHROPIC_API_KEY=sk-ant-...     # For Claude models
OPENAI_API_KEY=sk-...            # For GPT-4o models
DASHSCOPE_API_KEY=sk-...         # For Qwen VL models
```

You only need one. Claude Haiku 4.5 is the recommended default for quality and cost.

## Usage

### Web Interface

```bash
uv run uvicorn server:app --host 0.0.0.0 --port 9372
```

Open http://localhost:9372 in your browser. The web interface has two tabs:

- **Search** — Type to instantly search your photos with a responsive grid layout
- **Settings** — Add photos from a folder, choose provider/model, and manage the search index

### CLI

```bash
# Index a folder of photos
python indexer.py ./path/to/photos

# Use a specific provider
python indexer.py ./photos -p openai
python indexer.py ./photos -p qwen

# Use a different model
python indexer.py ./photos -m claude-sonnet-4-5-20250929

# Search from the command line
python indexer.py --search "beach sunset"

# Rebuild the search index (useful if search seems stale)
python indexer.py --rebuild-fts

# Delete a photo from the database
python indexer.py --delete "path/to/photo.jpg"

# Export database to JSON
python indexer.py --export

# Migrate a legacy JSON index to the database
python indexer.py --migrate
```

### CLI Options

```
python indexer.py [folder] [options]

Options:
  --db DB                   Database file (default: photo_index.db)
  --no-recursive            Process only top-level folder
  --delay SECONDS           Rate limit delay (default: 0.5)
  -p, --provider PROVIDER   AI provider: anthropic, openai, or qwen
  -m, --model MODEL         Model name
  --search QUERY            Search the database
  --delete FILEPATH         Delete a photo from the database
  --rebuild-fts             Rebuild the FTS search index
  --export                  Export database to JSON
  --migrate                 Migrate JSON index to SQLite
```

## AI Providers

| Provider | Default Model | Strengths | Cost (per 1K images) |
|----------|--------------|-----------|---------------------|
| Anthropic | claude-haiku-4-5 | Best location recognition, detailed descriptions | ~$5–$15 |
| OpenAI | gpt-4o-mini | Good general descriptions | ~$1.50–$4 |
| Qwen | qwen3-vl-flash | Cheapest option | ~$0.50–$2 |

## Project Structure

```
fastphoto/
├── indexer.py           # Core: image processing, CLI, progress tracking
├── db.py                # SQLite schema, FTS5, CRUD operations
├── server.py            # FastAPI backend + HTMX routes
├── templates/
│   ├── index.html       # Web UI (search + settings)
│   └── results.html     # Search results fragment
├── pyproject.toml       # Project dependencies
├── .env.example         # API key template
└── photos/              # Sample images
```

## How It Works

1. **Indexing** — Each image is sent to an AI vision model which returns structured metadata: description, location, tags, mood, etc.
2. **Storage** — Metadata is stored in SQLite with FTS5 full-text search for fast queries
3. **Search** — The web UI uses HTMX for instant, server-rendered search results with no JavaScript framework

## License

MIT
