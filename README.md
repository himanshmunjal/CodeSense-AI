# CodeSense — AI-Powered Codebase Intelligence Engine

[![CI](https://github.com/himanshmunjal/CodeSense-AI/actions/workflows/ci.yml/badge.svg)](https://github.com/himanshmunjal/CodeSense-AI/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)

> Point it at any GitHub repository. Ask it anything about the code.

CodeSense is a retrieval-augmented system for understanding codebases: it clones a repo, parses it into a real AST-derived structure (functions, classes, call graph) rather than flat text, indexes it with both semantic and structural embeddings, and answers natural-language questions with grounded, cited responses — *"Where is authentication handled?"*, *"What calls this function?"*, *"What are the different API routes?"*.

---

## Table of Contents

1. [What It Does](#what-it-does)
2. [Architecture](#architecture)
3. [Tech Stack](#tech-stack)
4. [Setup & Installation](#setup--installation)
5. [Usage](#usage)
6. [API Reference](#api-reference)
7. [Chunking Strategy](#chunking-strategy)
8. [Engineering Deep-Dives](#engineering-deep-dives)
9. [Project Structure](#project-structure)
10. [Known Limitations](#known-limitations)
11. [Roadmap](#roadmap)

---

## What It Does

| Capability | How |
|---|---|
| **Semantic search** | Ask in natural language, get the right function — not keyword matches |
| **Call graph traversal** | Find every caller of a function, recursively (impact analysis / blast radius) |
| **Grounded generation** | Every answer cites `file:function:line` — the LLM is instructed to answer only from retrieved context |
| **Query routing** | Classifies each question (lookup / relational / analytical / summarization) and picks the matching retrieval strategy |
| **Multi-language** | Python, JavaScript, TypeScript, Java, Go — real per-language AST parsing, not one generic regex pass |

### Query Types Supported

```
Lookup       → "Where is the payment processing function?"
               Strategy: semantic ANN search (Qdrant)

Relational   → "What calls this function?"
               Strategy: call graph traversal (networkx)

Analytical   → "Find inconsistent error handling patterns"
               Strategy: semantic search + hybrid scoring against the call graph

Summarization → "What are the different backend API routes?"
               Strategy: whole-file / module-level retrieval + synthesis
```

---

## Architecture

```
GitHub URL
    │
    ▼
┌─────────────────────────────────────┐
│           INGESTION LAYER           │
│  gitpython clone + PyGithub meta    │
│  File walker → language filter      │
│  Git diff → change detection        │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│            PARSING LAYER            │
│  tree-sitter AST                    │
│  (Python / JS / TS / Java / Go)     │
│  Entity extraction: functions,      │
│    classes, interfaces, enums,      │
│    records, imports, docstrings     │
│  Call graph → networkx DiGraph      │
│  Cyclomatic complexity per fn       │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│           EMBEDDING LAYER           │
│  Semantic: bge-small-en-v1.5        │
│    (384-dim, sentence-transformers) │
│  Structural: node2vec (128-dim)     │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│            INDEXING LAYER           │
│  Qdrant (two named vectors/point:   │
│    semantic + structural)           │
│  Payload: file_path, fn_name,       │
│    start_line, end_line, language,  │
│    complexity, docstring            │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│           RETRIEVAL LAYER           │
│  Query classifier (LLM call)        │
│  Semantic retriever (Qdrant ANN)    │
│  Graph retriever (networkx BFS)     │
│  Hybrid retriever (weighted blend)  │
│  Re-ranker (cross-encoder MiniLM)   │
└──────────────────┬──────────────────┘
                   │
                   ▼
┌─────────────────────────────────────┐
│          GENERATION LAYER           │
│  Groq-hosted LLM inference          │
│  Strict grounding system prompt     │
│  Confidence threshold gate          │
│  Structured output: answer +        │
│    sources + confidence + followups │
└──────────────────┬──────────────────┘
                   │
                   ▼
              Response
     (answer, file:fn:line citations,
      confidence, sources actually used,
      follow-up queries)
```

### Why Code as a Graph, Not Documents

Generic RAG over code treats a repository as a bag of text chunks. That breaks for code:

- A function is a semantic unit — splitting it mid-body destroys meaning
- Code has *relationships* — A calls B, C imports D, E extends F
- Some queries are purely structural — "what calls X" has nothing to do with text similarity

CodeSense represents every repository as two overlapping structures: a **semantic space** (embeddings of function/class bodies + docstrings) and a **call graph** (directed graph of call relationships from the AST). Which one a query uses — or how they're blended — depends on the query's classified type.

---

## Tech Stack

| Layer | Choice | Why |
|---|---|---|
| Parsing | `tree-sitter` | Real multi-language AST via grammar plugins, not regex |
| Graph | `networkx` | Call graph construction + traversal |
| Semantic embeddings | `BAAI/bge-small-en-v1.5` (via `sentence-transformers`) | See [Engineering Deep-Dives](#engineering-deep-dives) — this replaced an earlier choice for measured reasons |
| Structural embeddings | `node2vec` | Graph-neighborhood embeddings over the call graph |
| Vector DB | `Qdrant` | Multiple named vectors per point (semantic + structural together) |
| Re-ranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-attention re-scoring of the top-K candidates before generation |
| Cache / broker | `Redis` | Celery broker + result backend |
| Backend | `FastAPI` | Async routes, Pydantic validation |
| Async jobs | `Celery` | Non-blocking repo ingestion (a large repo can take minutes) |
| Frontend | `React` | Chat UI, source viewer, impact graph |
| LLM | Groq-hosted inference (`openai/gpt-oss-120b` by default) | OpenAI-compatible API, fast + free-tier inference — see note below |
| Evaluation | Custom Precision@5 harness (`evaluation/run_eval.py`) | Ground-truth file/function pairs against real repos |

**On the LLM choice:** the model is configured via `GROQ_MODEL` and swappable in one line (Groq exposes an OpenAI-compatible endpoint, called through the `openai` SDK). The original target model, `llama-3.1-70b-versatile`, was later decommissioned by Groq; the current default is `openai/gpt-oss-120b`. To compare models, run `evaluation/run_eval.py` against each — no cross-model numbers are claimed here.

---

## Setup & Installation

### Prerequisites

**Software:** Python 3.11+, Node.js 18+, Docker + Docker Compose, Git

**Accounts / API keys:**
- [Groq API key](https://console.groq.com) — free tier available
- [GitHub Personal Access Token](https://github.com/settings/tokens) — no scopes needed for public repos; for private repos use a fine-grained token with read-only *Contents* access

**Hardware:** 8GB RAM is workable but tight — see [Known Limitations](#known-limitations). 16GB+ recommended if you plan to ingest large repos (5,000+ chunks) or run anything else memory-heavy at the same time.

### 1 — Clone and configure

```bash
git clone https://github.com/himanshmunjal/CodeSense-AI.git
cd CodeSense-AI
cp .env.example .env
# Fill in GROQ_API_KEY and GITHUB_TOKEN
```

### 2 — Start infrastructure

```bash
docker compose up -d
# Starts Qdrant (port 6333) and Redis (host port = REDIS_PORT from .env, default 6379)
docker compose ps   # both should report healthy within ~10s
```

### 3 — Backend (two separate processes)

```bash
cd backend
python -m venv ../venv && source ../venv/bin/activate
pip install -r ../requirements.txt

# Terminal 1 — the API
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Terminal 2 — the ingestion worker (must be a separate process; see
# tasks/celery_worker.py's module docstring for why ingestion can't just
# run as a FastAPI background task)
celery -A tasks.celery_worker worker --loglevel=info --pool=solo
```

The embedding model (~130MB) and cross-encoder reranker download on first run, cached under `backend/.model_cache/`.

### 4 — Frontend

```bash
cd frontend
npm install
npm run dev
# http://localhost:5173
```

### 5 — Ingest a repository

```bash
curl -X POST http://localhost:8000/api/v1/ingest/ \
  -H "Content-Type: application/json" \
  -d '{"repo_url": "https://github.com/tiangolo/fastapi"}'
```

Ingestion runs asynchronously via Celery — poll `GET /api/v1/ingest/status/{task_id}` (returned in the response above), or watch it in the UI sidebar.

Re-ingesting a repo pulls its latest commit; if nothing changed since the last successful index it returns immediately, otherwise the index is rebuilt from scratch (pass `"force_reindex": true` to rebuild regardless).

### 6 — Run the tests

```bash
cd backend
pytest              # 118 unit + pipeline tests, no Docker or API keys needed
pytest -m integration   # extra tests against a live Qdrant (docker compose up -d)
```

---

## Usage

### Web UI

Three panels: repository list + ingest form (left), chat (center), source/impact viewer (right). Type a question about any ingested repo, or `impact analysis: <function_name>` to see its blast radius as a graph.

### API

```bash
curl -X POST http://localhost:8000/api/v1/query/ \
  -H "Content-Type: application/json" \
  -d '{
    "repo_url": "https://github.com/tiangolo/fastapi",
    "question": "Where is request validation handled?"
  }'
```

---

## API Reference

All routes are under `/api/v1/`. Every route file's own module docstring (`backend/api/routes/*.py`) has the full request/response schema with field-level explanations — this is a summary, not a substitute.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/v1/ingest/` | Queue async ingestion of a repo (`repo_url`, optional `branch`, `force_reindex`) |
| `GET` | `/api/v1/ingest/status/{task_id}` | Poll ingestion progress |
| `GET` | `/api/v1/ingest/repositories` | List indexed repositories |
| `DELETE` | `/api/v1/ingest/{owner}/{repo}` | Remove a repo's index |
| `POST` | `/api/v1/query/` | Ask a question (`repo_url`, `question`, optional `max_results`, `filter_language`, `filter_file_path`) |
| `POST` | `/api/v1/query/batch` | Multiple questions in one call (used by the eval harness) |
| `GET` | `/api/v1/query/history/{owner}/{repo}` | Past queries for a repo |
| `POST` | `/api/v1/impact/` | Blast-radius analysis for a function (`repo_url`, `function_name`, optional `file_path`, `max_depth`) |
| `POST` | `/api/v1/summarize/` | Summarize a file or directory (`repo_url`, `file_path` or `directory_path`, optional `focus_hint`) |
| `POST` | `/api/v1/summarize/module` | Summarize a module by path |
| `GET` | `/health` | Qdrant + Redis health (503 if degraded) |

**Example `POST /api/v1/query/` response shape:**

```json
{
  "question": "Where is request validation handled?",
  "answer": "Request validation happens in solve_dependencies()...",
  "sources": [
    {
      "file_path": "fastapi/dependencies/utils.py",
      "function_name": "solve_dependencies",
      "start_line": 42,
      "end_line": 67,
      "relevance_score": 0.74
    }
  ],
  "confidence": 0.74,
  "sources_used": 1,
  "query_type": "lookup",
  "followup_queries": ["..."],
  "latency_ms": 1820
}
```

`confidence` is the top retrieved chunk's raw embedding similarity — a retrieval-quality signal, not a claim about answer correctness. `sources_used` is a more direct grounding signal: how many of the returned `sources` the answer actually cites inline, vs. how many were retrieved. See [Engineering Deep-Dives](#engineering-deep-dives) for why both fields exist and what real bug led to adding the second one.

---

## Chunking Strategy

Most RAG-over-code tutorials split by token count (every N tokens, M overlap). That's wrong for code — it splits mid-function, orphaning a `raise` or a closing brace from the signature that gives it meaning. CodeSense instead parses each file's AST and treats each function, method, or class as one atomic chunk, regardless of token length. If a chunk is large, only the text sent to the *embedding* model is truncated (not the text sent to the *LLM* at generation time, which sees the real full body up to a shared prompt budget across all retrieved sources — see the module-chunk / snippet-budget notes in [Engineering Deep-Dives](#engineering-deep-dives)).

One deliberate addition beyond per-function chunking: each file also gets one **module-level chunk** covering the whole file, specifically so that top-level code that lives outside any function — database connection setup, app initialization, config constants — is searchable at all. That gap (and the retrieval-prompt bug it led to) is also covered below.

---

## Engineering Deep-Dives

Everything below was found by testing the system against real repositories and fixed with measured evidence — not assumed.

### The embedding model was fundamentally unsuited for the job

The original embedder (`microsoft/codebert-base`, used with hand-rolled mean-pooling) is not contrastively trained — it has no "pull similar things together, push dissimilar things apart" objective. Measured directly against this project's own indexed repos: the top 20 candidates for a real query all scored between **0.94 and 0.965 cosine similarity — including completely irrelevant code.** That's not a retrieval bug in the usual sense; it's a model that structurally cannot discriminate relevant from irrelevant content. Swapped to `BAAI/bge-small-en-v1.5` (via `sentence-transformers`, which applies each model's own correct pooling strategy automatically instead of assuming mean-pooling) and re-measured: a clearly irrelevant control query topped out around **0.43** similarity, while genuinely correct answers scored **0.55–0.79** — real separation. The confidence threshold (`RETRIEVAL_CONFIDENCE_THRESHOLD`) was then recalibrated from 0.65 to 0.5 to match this model's actual score distribution, since 0.65 had been silently refusing to answer correctly-retrieved questions.

### A dormant feature was actually dormant

`MetadataBuilder.build_module_chunk()` — code to index a whole file as a searchable unit, specifically for questions like "where is the database configured?" — existed in the codebase but was **never called** from the ingestion pipeline. Any code living outside a function (connection setup, app init, top-level constants) was invisible to every query, in every language, for every repo ever indexed. Wiring it in immediately fixed answers that had been confidently wrong ("the code reads from `collection`, which represents a data store" — because the actual `MongoClient(...)` call was never in any indexed chunk).

That fix then caused a real regression: a whole-file chunk has no upper bound on size, and once one was in the reranked top-5, a single query could request ~24,000 tokens against a provider with an 8,000-tokens-per-minute limit — a hard `413` error. The first fix (a flat per-source character cap) overcorrected: it truncated small files that never needed truncating, silently dropping real content (a 6-route API listed only 2 routes, because the flat cap cut the file off mid-way through). The actual fix is a **total prompt budget spent across all sources in rank order**, not a per-source cap — small results stay whole, only genuinely oversized combined context gets trimmed, and only as much as it needs to be.

### Multi-language parsing had real, silent gaps across every language

Systematic testing against large real-world repos (`gin-gonic/gin`, `google/gson`, `streamich/react-use`, `tiangolo/fastapi`) — not just the repo it was originally built against — surfaced entity-extraction bugs that had nothing to do with any one edge case:

- **JavaScript/TypeScript**: class `extends`/`implements` had never worked, for any class, since the code was written (a field-name lookup that didn't match the actual grammar). Abstract classes, interface method signatures, and interface `extends` were each simply never checked for. `const Foo = () => {}` — the dominant modern pattern for React components and hooks — produced zero indexed entities for an entire file style.
- **Java**: enums, Java 14+ records, and nested/inner classes (the Builder pattern, among others) were completely invisible to indexing. Javadoc comments were never extracted as documentation for any class or method, in any file, ever.
- **Go**: added language support from scratch this round (parsing, complexity analysis, doc-comment extraction, chunking). Interface/struct embedding wasn't tracked as inheritance, and generic receiver methods (`func (s *Stack[T]) Push(...)`) failed to link back to their type.
- **Python**: nested classes were invisible to indexing, same root cause as Java's.

Each of these was found by testing a real construct against a real file, verifying the fix with an isolated unit test, then confirming it end-to-end against the actual re-ingested repo and a live query — not assumed fixed after a code change.

### Evaluation methodology

`evaluation/run_eval.py` measures Precision@5 (does the ground-truth file/function appear in the top-5 returned sources) against hand-labeled query sets with real ground truth pulled from the live repo (`evaluation/eval_dataset_*.json`). Before the embedding-model swap and the multi-language fixes above, a baseline run against a small polyglot repo measured **37.5% Precision@5 overall, and 0% specifically on Go queries** (Go wasn't even being parsed at the time). Run `evaluation/run_eval.py --dataset <path>` yourself to reproduce current numbers against the current code — don't rely on a number frozen in a README that will drift as the code changes.

---

## Project Structure

```
codesense/
├── backend/
│   ├── ingestion/          # Clone, file walking, language filtering, git-diff change detection
│   ├── parsing/            # tree-sitter AST → entities (functions, classes, imports, call graph)
│   ├── embeddings/         # Semantic (sentence-transformers) + structural (node2vec)
│   ├── indexing/           # Qdrant collection setup, chunk schema, metadata building
│   ├── retrieval/          # Query classifier, semantic/graph/hybrid retrievers, reranker
│   ├── generation/         # Prompt construction, structured LLM output, response schema
│   ├── features/           # Impact analysis, inconsistency detection
│   ├── api/                # FastAPI app + routes (ingest, query, impact, summarize)
│   ├── tasks/               # Celery worker — the actual ingestion pipeline glue
│   ├── evaluation/          # eval_dataset*.json, run_eval.py, latency_benchmark.py
│   ├── tests/               # pytest suite (parser, retrieval, API, impact, ingestion pipeline)
│   └── config.py
│
├── frontend/
│   └── src/
│       ├── components/
│       │   ├── FileTree.jsx           # Repo list + ingest form
│       │   ├── ChatPanel.jsx          # Chat UI
│       │   ├── SourceViewer.jsx       # Click a citation, jump to the source
│       │   ├── ImpactGraph.jsx        # Blast-radius graph view
│       │   ├── SourcesUsedBadge.jsx   # Primary per-answer grounding indicator
│       │   └── ConfidenceBadge.jsx    # Per-source relevance score
│       └── App.jsx
│
├── docker-compose.yml       # Qdrant + Redis
├── .github/workflows/ci.yml # Backend tests + frontend build on every push
├── .env.example
└── requirements.txt
```

---

## Known Limitations

Honest, not aspirational — these are the real current boundaries:

| Limitation | Detail |
|---|---|
| **HTML/CSS unsupported** | The pipeline's chunking model is built around functions/classes; HTML elements and CSS selectors don't fit that shape without real design work, not just an extension-map entry |
| **8GB RAM is tight** | Docker's VM reservation plus two independent Python processes (API + worker), each loading their own copy of the embedding model, leaves little headroom. Running a large ingestion concurrently with any other heavy process has caused container crashes in practice |
| **Dynamic dispatch** | `obj.method()` calls where `obj`'s type isn't resolvable at parse time are missing from the call graph — a same-repo name-matching heuristic is used instead of real type inference |
| **Analytical query accuracy** | The weakest of the four query types — depends on hybrid scoring quality, which is harder to get right than pure lookup |
| **LLM provider rate limits** | Groq's free tier has strict per-minute token limits; a query with many large sources can still be tight against an 8,000 TPM cap even with the prompt-budget fix described above |
| **Retrieval quality varies by language** | The embedding model (`bge-small-en-v1.5`) is a general-purpose model, not code-specialized — measured directly, Go retrieval currently lags JS/Java on the same eval methodology. A code-specific embedding model is a real, unexplored option for improving this further |

---

## Roadmap

- [ ] Evaluate a code-specialized embedding model against the current general-purpose one, using the existing `run_eval.py` methodology (not a guess — see the deep-dive above on why a plausible-sounding swap turned out mixed in practice)
- [ ] Real type inference for dynamic-dispatch call graph edges
- [ ] Incremental graph updates (currently rebuilds the full call graph on re-ingest)
- [ ] HTML/CSS support (needs a genuinely different chunking model, not just a parser)
- [ ] VS Code extension

---

## Deploying

The defaults target local development. Before exposing the API anywhere public:

- Set `ENVIRONMENT=production` (disables `/docs` and `/openapi.json`) and `BACKEND_RELOAD=false`.
- Set `CORS_ORIGINS` to your frontend's URL, and `VITE_API_BASE_URL` (frontend `.env`) to the API's URL.
- Set `TRUST_PROXY_HEADERS=true` **only** if a reverse proxy sets `X-Forwarded-For`; otherwise rate limiting is keyed on the socket IP.
- Qdrant and Redis are published on `127.0.0.1` only and have no auth in `docker/` — add `QDRANT_API_KEY` / `REDIS_PASSWORD` if they move off-box.
- The API has no user authentication: anyone who can reach it can ingest/delete repos and spend your Groq quota. Put it behind auth (or a private network) if it's public.

---

## License

[MIT](LICENSE)
