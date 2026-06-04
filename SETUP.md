# Environment Variables — Setup Guide

Copy `.env.template` → `.env` and fill in the values below.

---

## Required

| Variable | Description | Example |
|---|---|---|
| `ANTHROPIC_API_KEY` | API key from console.anthropic.com — used for **both** Claude Vision and Voyage AI embeddings | `sk-ant-api03-…` |
| `NEO4J_PASSWORD` | Neo4j admin password (min 8 chars) | `MyStr0ng!Pass` |

## Optional (have defaults)

| Variable | Default | Description |
|---|---|---|
| `NEO4J_URI` | `bolt://localhost:7687` | Change to `bolt://neo4j:7687` inside Docker |
| `NEO4J_USER` | `neo4j` | Neo4j username |
| `NEO4J_DATABASE` | `neo4j` | Neo4j database name |
| `APP_ENV` | `development` | `development` or `production` |
| `APP_HOST` | `0.0.0.0` | Bind address |
| `APP_PORT` | `8000` | HTTP port |
| `APP_LOG_LEVEL` | `info` | `debug`, `info`, `warning`, `error` |
| `COSINE_SIMILARITY_THRESHOLD` | `0.75` | Min cosine similarity to consider a vector link (0–1) |
| `TEMPORAL_PROXIMITY_DAYS` | `90` | Captures within N days are candidates for temporal links |
| `LINK_STRENGTH_MIN` | `0.3` | Links below this strength are discarded |

---

## Anthropic API Key

1. Go to **https://console.anthropic.com/settings/keys**
2. Click **Create Key** → copy the `sk-ant-api03-…` value
3. Paste it as `ANTHROPIC_API_KEY=sk-ant-api03-…`

> The same key is used for both **Claude claude-opus-4-8** (vision extraction) and **Voyage AI** (embeddings, billed separately under your Anthropic account).

## Neo4j

### Docker (recommended)
The `docker-compose.yml` boots Neo4j automatically. Just set `NEO4J_PASSWORD`.

### Manual / Neo4j Desktop
1. Install Neo4j Desktop from https://neo4j.com/download
2. Create a project + local DBMS, set a password
3. Set `NEO4J_URI=bolt://localhost:7687` and `NEO4J_PASSWORD=<your password>`

### Neo4j Aura (cloud)
1. Sign up at https://neo4j.io/cloud/platform/aura-graph-database
2. Create a free instance, download credentials
3. Set `NEO4J_URI=neo4j+s://<instance>.databases.neo4j.io` and credentials

---

## Schema Initialisation

Run once after Neo4j is up:

```bash
python -m src.schema_setup
```

This creates:
- Uniqueness constraints on `Capture.id`, `Topic.name`, `Person.name`, `Place.name`
- Full-text index `capture_fulltext` (title + summary + raw_text)
- Range index `capture_created` (timestamp queries)
- Vector index `capture_embedding` (1024-dim cosine — requires Neo4j 5.x Enterprise **or** Community 5.13+)

> If the vector index creation fails (e.g., older Neo4j edition), the app still works — embedding similarity queries will simply return no results.
