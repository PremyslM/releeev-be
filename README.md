# Brain Graph — Knowledge Linking Backend

A FastAPI + Neo4j backend that turns images into a living knowledge graph.
Upload any screenshot, photo or document page — Claude claude-opus-4-8 extracts
structured attributes, Voyage AI generates an embedding, and the linking engine
automatically connects it to related past captures.

```
iOS App  ──POST /analyze-capture──▶  FastAPI
                                        │
                              ┌─────────┼──────────────┐
                              ▼         ▼              ▼
                        Claude API  Voyage AI      Neo4j Graph
                        (Vision)   (Embeddings)   (Cypher queries)
```

---

## Quick Start

### 1 — Clone & configure

```bash
cp .env.template .env
# Fill in ANTHROPIC_API_KEY and NEO4J_PASSWORD (see SETUP.md)
```

### 2 — Docker (recommended)

```bash
docker-compose up --build
```

Services:
| Service | URL |
|---|---|
| FastAPI | http://localhost:8000 |
| API Docs (Swagger) | http://localhost:8000/docs |
| Neo4j Browser | http://localhost:7474 |

### 3 — Local development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Start Neo4j separately (Docker or Desktop), then:
python -m src.schema_setup   # one-time schema init
python -m src.main           # hot-reload dev server
```

---

## API Reference

### `POST /analyze-capture`

Upload a base64-encoded image; get back extracted attributes + automatically linked captures.

**Request body**
```json
{
  "image_base64": "<base64-encoded JPEG/PNG/WebP>",
  "image_mime":   "image/jpeg"
}
```

**Response**
```json
{
  "capture_id":    "3f7e…",
  "attributes": {
    "title":        "iPhone 15 Pro Max Product Listing",
    "summary":      "Product page showing…",
    "raw_text":     "iPhone 15 Pro Max $1,199…",
    "capture_type": "photo",
    "topics":       ["apple", "iphone", "smartphone"],
    "people":       [],
    "places":       [],
    "language":     "en",
    "sentiment":    "positive",
    "confidence":   0.96
  },
  "embedding_dim": 1024,
  "linked_to": [
    {
      "capture_id": "a1b2…",
      "link_type":  "TOPIC_OVERLAP",
      "strength":   0.74,
      "title":      "Apple WWDC Notes",
      "summary":    "…"
    }
  ],
  "processing_ms": 1842
}
```

**Error codes**
| Status | Meaning |
|---|---|
| 401 | Invalid Anthropic API key |
| 422 | Validation error (invalid MIME type, missing field) |
| 429 | Claude API rate limit exceeded |
| 502 | Claude API / Voyage AI upstream error |
| 503 | Neo4j unavailable |

---

### `GET /captures/{id}`

Fetch a capture node and all its graph connections.

**Response**
```json
{
  "capture": { "id": "…", "attributes": {…}, "embedding": […] },
  "connections": [
    { "source_id": "…", "target_id": "…", "link_type": "EMBEDDING_SIMILARITY", "strength": 0.88 }
  ]
}
```

---

### `GET /search?q=…`

Full-text search across all captures.

**Query params**
| Param | Default | Description |
|---|---|---|
| `q` | required | Search query string |
| `limit` | 20 | Max results (1–100) |

**Response**
```json
{
  "query": "neural networks",
  "results": [
    { "capture_id": "…", "title": "…", "summary": "…", "score": 4.2, "created_at": "2025-01-15T…" }
  ],
  "total": 3
}
```

---

## Link Types

| Type | Signal | Threshold |
|---|---|---|
| `EMBEDDING_SIMILARITY` | Voyage AI cosine similarity | ≥ 0.75 (configurable) |
| `TOPIC_OVERLAP` | Shared topic keywords | ≥ 1 shared topic |
| `ENTITY_MATCH` | Same person or place | ≥ 1 shared entity |
| `TEMPORAL_PROXIMITY` | Created within 90 days | < 90 days apart |

---

## Project Structure

```
second-brain/
├── src/
│   ├── config.py           # Pydantic settings (reads .env)
│   ├── models.py           # Pydantic data models
│   ├── neo4j_driver.py     # Async Neo4j connection pool
│   ├── schema_setup.py     # Constraints + indexes bootstrap
│   ├── vision_extractor.py # Claude Vision + Voyage AI embeddings
│   ├── queries.py          # Cypher query library
│   ├── link_engine.py      # Similarity math + graph queries
│   └── main.py             # FastAPI app
├── tests/
│   ├── test_extraction.py  # 8 tests — vision extraction + embedding
│   └── test_linking.py     # 9 tests — cosine math + link engine
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.template
├── SETUP.md
└── README.md
```

---

## Running Tests

```bash
pytest tests/ -v
# or with coverage:
pytest tests/ -v --tb=short
```

All tests use mocks — no live API key or running Neo4j required.

---

## iOS Integration

The API is ready to consume from Swift / Objective-C:

1. Capture screen → encode as base64 JPEG
2. `POST /analyze-capture` with `Content-Type: application/json`
3. Display `attributes` + `linked_to` in your UI
4. Use `GET /captures/{id}` to show the full knowledge graph for a node
5. Use `GET /search?q=…` for search-as-you-type

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM Vision | Claude claude-opus-4-8 (Anthropic) |
| Embeddings | Voyage AI `voyage-3` (via Anthropic) |
| Graph DB | Neo4j 5.x Community |
| API | FastAPI + Uvicorn |
| Validation | Pydantic v2 |
| Containers | Docker + Docker Compose |
