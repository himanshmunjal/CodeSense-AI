# CodeSense — Architecture Notes & Design Decisions

## 1. What "Understanding Code" Means in This System

CodeSense is not a text search engine over code files. It understands code at two levels simultaneously:

- **Semantic level** — what does this code *mean*? (embeddings over function/class bodies + docstrings)
- **Structural level** — how does this code *connect*? (AST-derived call graph)

Every architectural decision below flows from that distinction.

---

## 2. Language Scope

| Language   | Tree-sitter Grammar    | Notes |
|------------|-------------------------|-------|
| Python     | tree-sitter-python      | Functions, classes (incl. nested), decorators, async, match statements |
| JavaScript | tree-sitter-javascript  | Function declarations, `const Foo = () => {}` (the dominant modern pattern), classes incl. `extends`/`implements` |
| TypeScript | tree-sitter-typescript  | JS superset + interfaces (methods, `extends`), abstract classes, generics |
| Java       | tree-sitter-java        | Classes, interfaces, enums, records (Java 14+), nested/inner classes, Javadoc |
| Go         | tree-sitter-go          | Functions, receiver methods (incl. generics), struct/interface embedding, doc comments |

Excluded: Rust, C++, Ruby, HTML/CSS. HTML/CSS specifically don't fit the current chunking model (built around functions/classes as the atomic unit) without real design work — see the README's Known Limitations.

---

## 3. LLM Provider: Groq-Hosted Inference

Generation and query classification go through Groq's OpenAI-compatible API (`groq_base_url` in `config.py` points the `openai` SDK client at Groq instead of OpenAI — that's the only code-level difference from calling OpenAI directly). The specific model is configured via `GROQ_MODEL` and swappable in one line.

**Note on model history:** the original target model, `llama-3.1-70b-versatile`, was decommissioned by Groq after this project's initial development; the current default is `openai/gpt-oss-120b`. If you want a real quality/latency/cost comparison between candidate models for a resume writeup, run `evaluation/run_eval.py` against each and report what you actually measure — a specific benchmark table lived in an earlier version of this document without ever having been produced by running anything, which is worth knowing if you're basing career claims on this repo's history.

**Embeddings are fully local, independent of the LLM provider** — see §7.

---

## 4. GitHub Ingestion Strategy

**Decision: clone locally via gitpython, not GitHub API for file content.**

- GitHub API rate limits (5,000 req/hr authenticated) get exhausted fast fetching file contents individually on a 1,000-file repo.
- A local clone gives full git history, enabling diff-based change detection on re-ingestion.
- PyGithub is used only for metadata (description, default branch, stars).

**Repo size limits:** warns at 50,000 files, hard stop at 100,000. Sharding not implemented.

---

## 5. Parsing Strategy: Why Tree-sitter

| Approach | Why rejected |
|---|---|
| Regex | Can't handle nested structures or multiline constructs |
| Naive token-count chunking | Splits mid-function, destroying the semantic unit |
| LLM-based chunking | Too slow/expensive at ingestion scale |
| Per-language AST libs (`ast`, `javalang`, ...) | One library per language, high maintenance burden |

Tree-sitter gives one library across languages via grammar plugins, a real AST, and error recovery (parses syntactically broken code, which real repos contain).

**What gets extracted per file** (language-dependent, see §2): functions/methods with signature, docstring, line range, and body; classes/interfaces/enums/records including nested ones and inheritance relationships; imports; a per-file module-level chunk covering top-level code that lives outside any function (see §6 — this exists specifically because it was *missing* for a while and that was a real, found bug, not a day-one design decision).

**Chunking unit:** a function/method/class is one atomic chunk, never split mid-body. The text sent to the *embedding* model is truncated for long chunks (currently 1,500 chars — see `indexing/chunk_schema.py` for why that number and not something smaller); the text sent to the *LLM at generation time* is governed separately by a total-prompt character budget spread across all retrieved sources, not truncated per-chunk (see `api/routes/query.py::_apply_snippet_budget` — this two-stage design exists because a flat per-source cap either broke large repos or truncated small ones depending which way it was tuned; a shared total budget spent in rank order is what actually works for both).

---

## 6. Full Pipeline

```
GitHub URL
    │
    ▼
[ Ingestion Layer ]
  gitpython clone → PyGithub metadata
  File walker → language filter
  Git diff → change detection on re-ingest
    │
    ▼
[ Parsing Layer ]
  tree-sitter AST per file (Python / JS / TS / Java / Go)
  Entity extraction: functions, classes, interfaces, enums,
    records, nested types, imports, docstrings
  Call graph construction: networkx DiGraph
  Cyclomatic complexity per function (from AST)
    │
    ▼
[ Embedding Layer ]
  Semantic:    bge-small-en-v1.5 (function/class body + docstring) → 384-dim vector
  Structural:  node2vec on call graph nodes → 128-dim vector
    │
    ▼
[ Indexing Layer ]
  Qdrant: one collection per repository
  Two named vectors per point: { semantic: 384-dim, structural: 128-dim }
  Payload per point:
    { file_path, function_name, start_line, end_line,
      language, complexity, docstring, source_code }
    │
    ▼
[ Query Layer ]
  1. Classify query type → LLM call (lookup/relational/analytical/summarization)
  2. Route to retrieval strategy:
       Lookup / Summarization → Qdrant ANN semantic search
       Relational             → networkx traversal on call graph
       Analytical             → hybrid semantic + structural-centrality scoring
  3. Re-rank top-K with cross-encoder/ms-marco-MiniLM-L-6-v2
    │
    ▼
[ Generation Layer ]
  Groq-hosted LLM call
  Strict grounding system prompt — answer only from retrieved context
  Confidence gate: below settings.retrieval_confidence_threshold → refuse
  Structured output: { answer, sources, confidence, sources_used, followup_queries }
    │
    ▼
[ Response ]
  Answer with file:function:line citations
  confidence (retrieval similarity) + sources_used (how many sources the
    answer actually cited — see §10 for why both exist)
  Suggested follow-up queries
```

---

## 7. Embedding Design

### Semantic Embeddings

**Model: `BAAI/bge-small-en-v1.5`, 384-dim, loaded via `sentence-transformers`.**

This replaced an earlier choice, `microsoft/codebert-base`, for a measured reason, not a guess: CodeBERT is not contrastively trained (no "similar things close, dissimilar things far" objective), and its raw mean-pooled embeddings measured as effectively unable to discriminate relevant from irrelevant code on this project's own indexed repos — the top 20 candidates for a real query all landed in a **0.94–0.965** cosine similarity band regardless of actual relevance. bge-small, re-measured the same way, showed real separation: an irrelevant control query topped out around **0.43**, genuinely correct answers scored **0.55–0.79**.

Loading via `sentence-transformers.SentenceTransformer` (rather than hand-rolling `AutoModel` + manual pooling, which this project also used to do) matters beyond convenience: it applies whatever pooling strategy a given model was actually trained with (mean-pooling vs. CLS-token, different dimensionality) automatically, from that model's own published config. Hand-rolling pooling hard-codes an assumption specific to one model — get it wrong for the next model you try and embeddings are silently corrupted with no error, not a crash you'd notice.

`RETRIEVAL_CONFIDENCE_THRESHOLD` (the hard refuse-if-below gate) is tied to whichever embedding model is configured — it was recalibrated from 0.65 to 0.5 alongside the model swap, using the measured 0.43/0.55 numbers above as the calibration points. Re-measure before changing the embedding model again; don't carry the threshold forward assuming it still applies.

### Structural Embeddings

**Model: node2vec on the call graph.**

Each function is a node; a directed edge means "A calls B." node2vec's biased random walks produce embeddings where structurally-adjacent functions end up similar. Fusion at retrieval time for analytical queries:

```
hybrid_score = semantic_weight * semantic_similarity + structural_weight * structural_centrality
```

Default weights (`HYBRID_SEMANTIC_WEIGHT` / `HYBRID_STRUCTURAL_WEIGHT` in `.env`) are 0.7/0.3.

---

## 8. Vector DB: Why Qdrant

| Feature | Qdrant | ChromaDB | Pinecone |
|---|---|---|---|
| Filtered search | Native | Limited | Yes |
| Self-hostable | Yes | Yes | No |
| Multiple named vectors per point | Yes | No | No |
| Cost | Free (self-hosted) | Free | Paid |

The deciding feature is multiple named vectors per point — semantic (384-dim) and structural (128-dim) live on the same point, searchable independently or blended. Collection naming: `codesense_{owner}_{repo}` (sanitized, lowercase, hyphens → underscores).

---

## 9. Query Classification & Routing

| Query Type | Example | Strategy |
|---|---|---|
| Lookup | "Where is the payment function?" | Semantic ANN search |
| Relational | "What calls this function?" | Graph traversal from the target node |
| Analytical | "Find inconsistent error handling" | Hybrid semantic + structural-centrality scoring |
| Summarization | "What are the different API routes?" | Semantic search over module-level chunks |

Classification is a single LLM call with structured JSON output — adds latency but prevents routing a structural question ("what calls X") to a retriever that has no concept of call relationships, which would silently return irrelevant results.

---

## 10. Generation Grounding Rules

1. Every answer cites `file_path:function_name:Lstart-Lend` inline — the system prompt mandates this format.
2. If the top reranked source's bounded similarity score (`hybrid_score`, **not** the cross-encoder's raw unbounded logit — see `retrieval/reranker.py`) is below `settings.retrieval_confidence_threshold`, the system refuses rather than generating from weak context.
3. The system prompt forbids answering from the model's own prior knowledge of frameworks/libraries — only the retrieved context window.
4. Response schema enforced via the LLM's JSON mode + Pydantic validation: `{ answer, sources, confidence, sources_used, followup_queries }`.

**Why both `confidence` and `sources_used` exist:** `confidence` is the top source's raw embedding similarity — a retrieval-quality signal about what was *found*, not a claim about the generated answer's correctness. A fully correct, complete answer can legitimately score in the 55-60% range with a general-purpose embedding model, which reads as "low confidence" if that's the only number shown. `sources_used` — how many of the returned sources the answer's own citations actually reference — is a more direct grounding signal that isn't dependent on a particular model's score calibration. Both are returned; the frontend leads with `sources_used`.

---

## 11. Known Limitations

See the README's [Known Limitations](README.md#known-limitations) section — kept in one place rather than duplicated (and allowed to drift) across two documents.
