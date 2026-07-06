# Deployment & Environments

How the Brain Graph backend is organized across environments, and how to move
it to a new server or database with minimal fuss.

The golden rule: **all environment-specific config lives in environment
variables** (12-factor). Nothing is hardcoded. Migrating to a new server or DB
is "set the env vars, run one bootstrap command, deploy."

---

## 1. Environments

| Environment     | Git branch    | Backend                  | Database              | iOS build |
| --------------- | ------------- | ------------------------ | --------------------- | --------- |
| **development** | `development` | `docker compose up`      | local Docker Neo4j    | Debug     |
| **production**  | `main`        | Vercel (`releeev-be`)    | Neo4j Aura            | Release   |
| *testing* (later) | `testing`   | Vercel Preview           | 2nd Aura / shared     | —         |

- **development**: everything local. The iOS Debug build points at
  `http://localhost:8000` (Simulator) — on a physical device set the Mac's LAN IP
  in the app's Server settings.
- **production**: `main` deploys to Vercel Production, backed by Aura. The iOS
  Release build points at `https://releeev-be.vercel.app`.

---

## 2. Environment-variable contract

Every backend reads the same variables (see `src/config.py`). `.env.template` is
the canonical list — copy it to `.env` for local dev.

| Variable             | Dev (local `.env`)          | Prod (Vercel env vars)                    |
| -------------------- | --------------------------- | ----------------------------------------- |
| `ANTHROPIC_API_KEY`  | your key                    | your key                                  |
| `VOYAGE_API_KEY`     | your key                    | your key                                  |
| `NEO4J_URI`          | `bolt://localhost:7687`     | `neo4j+s://<id>.databases.neo4j.io`       |
| `NEO4J_USER`         | `neo4j`                     | Aura username (e.g. the instance id)      |
| `NEO4J_PASSWORD`     | local password              | Aura password                             |
| `NEO4J_DATABASE`     | `neo4j`                     | Aura database (e.g. the instance id)      |
| `JWT_SECRET`         | anything                    | **strong random, ≥16 chars**              |
| `JWT_TTL_DAYS`       | `30`                        | `30`                                      |
| `APP_ENV`            | `development`               | `production`                              |
| `RUN_SCHEMA_SETUP`   | `true`                      | `false` (schema created once, see §5)     |

> **Production safety net:** with `APP_ENV=production`, the app **refuses to
> boot** if `JWT_SECRET` is missing/weak (`src/config.py → safety_problems`).
> Better down than serving auth with the dev secret.

> **Aura gotcha:** the credentials file Aura gives you uses `NEO4J_USERNAME`, but
> the app reads `NEO4J_USER`. Some Aura Free instances use the **instance id** as
> both username and database name — not `neo4j`. Copy them into `NEO4J_USER` /
> `NEO4J_DATABASE` explicitly. See `memory` note *local-neo4j-config-drift*.

Generate a strong secret:
```bash
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

---

## 3. Local development

```bash
cp .env.template .env          # fill in keys; local NEO4J_* already match Docker
docker compose up -d           # API on :8000, Neo4j on :7687/:7474
curl localhost:8000/health     # {"status":"ok","env":"development"}
```

Run the tests (mocked — no live services needed):
```bash
pytest -q
```

---

## 4. Vercel + GitHub Actions (production)

Deploys are driven by **GitHub Actions** (`.github/workflows/deploy.yml`):
tests run on every push/PR; on a push, the app is deployed **only if tests pass**
— `development` → Preview, `main` → Production.

**One-time setup:**

1. **Link the repo to Vercel** (creates the project & ids):
   ```bash
   npm i -g vercel && vercel link
   cat .vercel/project.json   # → orgId + projectId
   ```
2. **Add GitHub repo secrets** (Settings → Secrets and variables → Actions).
   These are secrets — set them in the GitHub UI, never commit their values here:
   - `VERCEL_TOKEN` — from https://vercel.com/account/tokens
   - `VERCEL_ORG_ID` — from `.vercel/project.json` (the `orgId` field)
   - `VERCEL_PROJECT_ID` — from `.vercel/project.json` (the `projectId` field)
3. **Add Vercel env vars** (Vercel dashboard → Settings → Environment Variables),
   scoped to **Production** (and later **Preview**): the full §2 contract with the
   Aura values, `APP_ENV=production`, `RUN_SCHEMA_SETUP=false`.
4. **Disable Deployment Protection** (Settings → Deployment Protection → Vercel
   Authentication → Disabled) so the API is publicly reachable — otherwise every
   request gets a 302 → the app can't read JSON.
5. Push to `main` → Actions runs tests → deploys to `releeev-be.vercel.app`.

> If you'd rather not use Actions, Vercel's native Git integration also works
> (`main` → Production automatically). Pick one to avoid double deploys.

---

## 5. Migrating to a new server or database

The whole point of the env-var discipline: this is a short checklist.

**New database (e.g. a fresh Aura instance or self-hosted Neo4j):**
1. Point config at it: set `NEO4J_URI` / `NEO4J_USER` / `NEO4J_PASSWORD` /
   `NEO4J_DATABASE` (in `.env` locally, or Vercel env vars for prod).
2. Create the schema (constraints + vector/fulltext indexes) — **one command**:
   ```bash
   python -m src.schema_setup
   ```
   Idempotent (`IF NOT EXISTS`), safe to re-run.
3. (Optional) move data — see below.
4. Restart / redeploy. Done.

**New backend host (e.g. Railway/Render/Fly instead of Vercel):**
- The app is a standard ASGI app (`src.main:app`) + `Dockerfile`. Any container
  host works: set the same env vars, expose `$APP_PORT`, run
  `uvicorn src.main:app`. No Vercel-specific code. Long-running hosts also avoid
  the serverless cold-start + `maxDuration` limits, and run `lifespan` normally
  (so `RUN_SCHEMA_SETUP=true` bootstraps on first boot).
- Update the iOS Release URL (`APIConfig.defaultBaseURL`) and bump
  `APIConfig.configVersion` so devices pick it up.

**Moving data between databases** (Neo4j → Neo4j):
- Aura → Aura: use the Aura console snapshot/restore, or `neo4j-admin database dump`.
- Cypher export/import for small datasets, or APOC `apoc.export.cypher.all`.
- Because all captures carry `user_id`, data stays correctly per-user after a move.

---

## 6. iOS environment mapping

`APIConfig.swift` selects the backend at **compile time** by build configuration:

- **Debug** build → `http://localhost:8000` (override with LAN IP on device)
- **Release** build → `https://releeev-be.vercel.app`

A persisted URL in the app's settings overrides the compile-time default; the
one-time `migrateDefaultsIfNeeded()` (bumped via `configVersion`) clears stale
values after a default change, so a new build "just works" without manual reset.
