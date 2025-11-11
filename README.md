# Mistral EvalOps Workbench

> RAG with citations + versioned prompts + dataset-driven A/B evaluations + run-level observability (latency, tokens, cost, context).
> Stack: Next.js (TS) • FastAPI (Py) • Postgres + pgvector • Mistral SDK • Optional vLLM.

[![Status](https://img.shields.io/badge/status-MVP-green)](#)
[![Stack](https://img.shields.io/badge/stack-Next.js%20%7C%20FastAPI%20%7C%20Postgres%20%7C%20pgvector-blue)](#)
[![LLM](https://img.shields.io/badge/LLM-Mistral--small--latest%20%7C%20mistral--embed-purple)](#)
[![Docker](https://img.shields.io/badge/run-docker--compose-informational)](#)

Why this repo exists (maps to the role):
- Chatbot, search engines, document answering → RAG with inline citations.
- Dashboards, evaluation interfaces → A/B evals and Runs dashboard with p50/p95 latency, tokens, cost.
- Developer experience → versioned prompts, quickstart in 3 commands, and a clean API surface.

---

## Features

- [x] Grounded chat over uploaded docs with inline citations `[n]`.
- [x] Streaming (SSE) responses.
- [x] Prompt registry (versioned system/user templates; diff).
- [x] Run logging (model, backend, tokens, latency; stored in DB).
- [ ] pgvector retrieval (cosine) - MVP uses fallback, pgvector landing shortly.
- [ ] A/B evaluations (LLM-as-judge: relevance, groundedness) with JSON mode.
- [ ] Batch mode for evals.
- [ ] OpenTelemetry traces.

If you are reviewing quickly: open /chat, ask once, then check /runs for tokens and latency. Diff prompts in /prompts.

---

## Architecture

```text
apps/
  api/  # FastAPI (Python)
    app/
      routers/ chat|ingest|eval|metrics
      rag/ chunker|embeddings|retriever
      evals/ judge (JSON-mode)
      core/ settings|db|mistral_client|logging
      schemas.py (Pydantic I/O)
      main.py
  web/  # Next.js (TypeScript)
    app/ chat|prompts|datasets|runs|layout
    components ChatUI|PromptDiff|RunsTable|UploadBox
    lib/ api.ts (SSE/json), env.ts
infra/
  db/init.sql  # extensions + tables (pgvector: 1024 dims)
  docker-compose.yml  # db + api + web
packages/
  prompts/ rag_v1.json, rag_v2.json
  evals/ judge_schema.json, judge_prompt.txt
datasets/
  docs/  # seed PDFs
  qa.jsonl  # 30–50 Q/A for evals
```

Key tables (Postgres): documents, chunks(vector(1024)), prompts, runs, judgements, datasets, dataset_items.
infra/db/init.sql bootstraps schema and pgvector.

---

## Quickstart (3 commands)

Prereqs: Docker Desktop 4.27+, a Mistral API key.

```bash
cp .env.example .env            # add your MISTRAL_API_KEY
cd infra
docker compose up --build -d    # launches Postgres+pgvector, API, and Web
```

Open:
- Web http://localhost:3000
- API http://localhost:8001
- DB localhost:5432

Stop stack:

```bash
docker compose down -v
```

Configuration (.env)

```
# LLM
MISTRAL_API_KEY=replace_me
BACKEND=mistral                     # or vllm
LLM_BASE_URL=https://api.mistral.ai
MODEL_CHAT=mistral-small-latest
MODEL_EMBED=mistral-embed          # 1024 dims

# DB
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=workbench
POSTGRES_HOST=db
POSTGRES_PORT=5432
DATABASE_URL=postgresql://postgres:postgres@db:5432/workbench

# Web
NEXT_PUBLIC_API_BASE_URL=http://localhost:8001
```

Flip to local vLLM: set BACKEND=vllm and LLM_BASE_URL=http://host.docker.internal:8000/v1 then docker compose restart api.

---

## API (contract)

Base: http://localhost:8001

| Method | Path | Purpose | In (shape) | Out (shape) |
|---|---|---|---|---|
| POST | /ingest | Upload PDFs → chunk → embed → store | multipart/form-data: files[] | {document_ids[], chunks} |
| POST | /ingest/retrieve | Probe retrieval on a query | {query, k} | {query, results:[{chunk_id,filename,score,snippet}]} |
| POST | /chat/complete | RAG answer (non-streaming) | {query, retriever:{top_k}, prompt:{name,version}} | {run_id, answer, context[], tokens{}, latency_ms} |
| POST | /chat/stream | RAG answer (SSE) | same as above | data: {"citation":...} / {"delta":"..."} |
| GET | /prompts | List prompt versions | - | [{name,version,...}] |
| GET | /prompts/diff | Diff two versions | ?name=&a=&b= | {systemDiff,userTemplateDiff,paramsDiff} |
| GET | /runs | Paginated runs with aggregates | ?limit=&offset= | [{id,kind,model,p50,p95,tokens,cost,...}] |
| GET | /runs/{id} | Run detail | - | {config,context[],judgements?} |
| POST | /eval/run | A/B eval over dataset | {dataset_id, variant_a, variant_b} | {group_id, run_ids[]} |

OpenAPI is auto-served at /openapi.json (AI-readable).

---

## Data: Dataset schema (datasets/qa.jsonl)

Each line:

```json
{"question":"...", "reference_answer":"...", "doc_ids":["...","..."]}
```

Used for batch evals (LLM-as-judge returns {relevance, groundedness, rationale} in JSON mode).

---

## How it works (MVP → full)

### Ingest → Chunk → Embed

- Extract text (PyPDF); sliding window (size=1000, overlap=150).
- Embeddings via mistral-embed (1024). Stored in chunks.embedding.
- If pgvector unavailable, a Python cosine fallback is used (slower).

### Retrieve → Answer → Cite

- Embed query once; top-k by cosine.
- Build numbered context; prompt enforces [n] citations or "Not in context."
- Stream tokens via SSE; log run with tokens, latency, and cost.

### A/B + Judge (coming up next)

- Variants: prompt versions and retriever params.
- Judge returns strict JSON (relevance, groundedness, rationale).
- Group A/B runs; show deltas and regressions in /runs.

### Observability

- Every run persists: model, backend, prompt(name,version), retriever.k, latency_ms (and p50/p95 rollups), tokens_in/out, cost_cents, context[] (with chunk/file and score).
- Planned: OpenTelemetry traces (web → api → llm).

### Security

- No secrets committed; .env is git-ignored; .env.example documents required keys.
- Request/response logs avoid storing user secrets or raw PDFs (only chunk text and file names).
- Optional moderation/redaction hook before persist (planned).


### Make targets (DX)

```
up:        ## build & run all services
	cd infra && docker compose up --build -d
down:      ## stop & clean
	cd infra && docker compose down -v
logs:
	cd infra && docker compose logs -f --tail=200
seed:
	cd infra && docker compose exec api python -m app.scripts.seed_db
eval-demo:
	cd infra && docker compose exec api python -m app.scripts.run_eval datasets/qa.jsonl
```

---

## Roadmap (short and honest)

- Land pgvector path as default (keep Python fallback).
- Wire LLM-as-judge (JSON-mode) and A/B deltas in /runs.
- Add reranker toggle and simple metrics/summary cards.
- OTel traces; seed Batch API path for evals at scale.
- Record a 3-min demo video (setup → chat → A/B → eval).

---

## Author note (context)

This project was vibe-coded under tight time constraints during university midterm exams. I prioritized a working MVP with a turnkey setup and a clear path to evals and observability, which matches Mistral’s emphasis on dashboards and evaluation interfaces (not just "chat"). I’m actively expanding the pgvector path and A/B judging next because measured improvement beats feature breadth. Happy to iterate fast and push this to production-grade quality.

Thabet Selim

---