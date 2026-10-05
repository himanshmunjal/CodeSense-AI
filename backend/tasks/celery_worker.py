"""
tasks/celery_worker.py — Async Ingestion Pipeline
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHY THIS FILE EXISTS
────────────────────
api/routes/ingest.py dispatches `ingest_repository_task.delay(...)` and polls
`celery_app` for status — this is the module that provides both. It is the
one place in the codebase where every pipeline stage described in
Architecture-notes.md §6 ("Full Pipeline") is actually wired together end to
end: clone → walk → parse → extract entities → build call graph → embed
(semantic + structural) → build chunk metadata → index into Qdrant.

Runs as a separate OS process from the FastAPI app:
    celery -A tasks.celery_worker worker --loglevel=info

WHY A SEPARATE PROCESS (not a FastAPI background task):
    Ingestion loads the embedding model and runs CPU-bound tree-sitter parsing
    over potentially thousands of files — this would block the FastAPI event
    loop for minutes. Celery workers run in their own process(es), so the API
    stays responsive to /query requests on other repos while ingestion runs.

RE-INGESTION:
    An existing clone is fast-forwarded to the latest commit. If HEAD matches
    the last successfully indexed commit the task returns immediately;
    otherwise the repo is rebuilt from scratch (call graph, embeddings and
    Qdrant collection) — see step 3 below for why it isn't patched per-file.

KNOWN GAPS (flagged rather than silently worked around):
    - Call-edge resolution: entity_extractor.py's FunctionEntity.calls field is
      declared "populated downstream" but no module actually populates it.
      This file does it with a simple same-repo name-matching heuristic
      (_resolve_call_edges below) — it will miss dynamic dispatch
      (`obj.method()`) per the documented v1 limitation (README "Known
      Limitations"), but resolves plain `function_name(...)` calls.
    - metadata_builder.MetadataBuilder.enrich_with_call_graph() looks up nodes
      by "{file_path}::{fully_qualified_name}", matching this file's and
      retrieval/graph_retriever.py's node_id convention, so in_degree/
      out_degree end up populated correctly on chunks.
    - Per-function cyclomatic complexity needs a tree-sitter AST node
      (complexity_analyzer.ComplexityAnalyzer.analyze_function), but
      entity_extractor.py's FunctionEntity only carries body_text, not the
      node. Re-locating the exact node per function is not worth the
      complexity here, so this file counts decision-point keywords directly
      from body_text (_count_decision_points) and feeds that into
      MetadataBuilder's existing McCabe formula (decision_points + 1).
"""

from __future__ import annotations

import pickle
import re
import time
from pathlib import Path
from typing import Optional

from celery import Celery
from loguru import logger

from config import settings

from ingestion.github_loader import clone_repository, get_repo_metadata
from ingestion.file_walker import walk_repository, SourceFile
from parsing.change_detector import ChangeDetector
from parsing.tree_sitter_parser import parse_files_batch
from parsing.entity_extractor import extract_entities_batch, FileEntities, FunctionEntity
from parsing.call_graph_builder import CallGraphBuilder, FunctionNode, CallEdge
from embeddings.code_embedder import CodeEmbedder
from embeddings.graph_embedder import GraphEmbedder
from indexing.metadata_builder import MetadataBuilder
from indexing.chunk_schema import Language, CodeChunk
from indexing.qdrant_client import QdrantIndexer

import redis as redis_lib
import json


# ─────────────────────────────────────────────────────────────────────────────
# Celery App
# ─────────────────────────────────────────────────────────────────────────────

celery_app = Celery(
    "codesense",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)

# task_track_started: without this, Celery jumps straight from PENDING to
# SUCCESS/FAILURE — api/routes/ingest.py's status endpoint relies on the
# intermediate STARTED state (with progress in task.info) to report progress.
celery_app.conf.update(
    task_track_started=True,
    worker_prefetch_multiplier=1,   # one ingestion job per worker at a time — these are heavy
    task_acks_late=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — heuristic call resolution (see module docstring, "KNOWN GAPS")
# ─────────────────────────────────────────────────────────────────────────────

_CALL_PATTERN = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")

# Definitions also look like `name(` — `def __call__(self)` inside a nested
# class would otherwise count as a CALL to every `__call__` in the repo.
# Stripped before call matching: Python def/class, JS/TS function, Go func
# (including `func (r *T) Name(` methods).
_DEFINITION_PATTERN = re.compile(
    r"\b(?:def|class|function\*?)\s+[A-Za-z_][A-Za-z0-9_]*"
    r"|\bfunc\s*(?:\([^)]*\)\s*)?[A-Za-z_][A-Za-z0-9_]*"
)

# Python/JS/TS/Java keywords that look like calls but aren't user functions.
_CALL_KEYWORDS = {
    "if", "for", "while", "switch", "catch", "return", "print", "super",
    "new", "async", "await", "yield", "with", "except", "elif",
    "func", "function", "def", "lambda",
}


def _resolve_call_edges(
    all_functions: dict[str, tuple[FunctionEntity, str]],
) -> list[CallEdge]:
    """
    Best-effort call graph edges from plain `name(...)` call sites.

    all_functions maps qualified_name -> (FunctionEntity, node_id) for every
    function extracted across the whole repo. For each function's body_text,
    finds identifier(...) call sites and links to any other function in the
    repo sharing that bare name. Ambiguous (multiple functions with the same
    name in different files) — resolves to all of them, since we cannot
    determine which one is called without full type inference (out of scope
    for v1, see README "Known Limitations" — dynamic dispatch).
    """
    by_bare_name: dict[str, list[str]] = {}
    for qualified_name, (entity, _node_id) in all_functions.items():
        by_bare_name.setdefault(entity.name, []).append(qualified_name)

    edges: list[CallEdge] = []
    for qualified_name, (entity, node_id) in all_functions.items():
        body_without_definitions = _DEFINITION_PATTERN.sub(" ", entity.body_text)
        called_names = set(_CALL_PATTERN.findall(body_without_definitions)) - _CALL_KEYWORDS
        called_names.discard(entity.name)  # skip self-recursion noise in edge count, still allowed below
        for called_name in called_names:
            for callee_qualified_name in by_bare_name.get(called_name, []):
                if callee_qualified_name == qualified_name:
                    continue  # recursion — not useful for blast-radius ranking
                edges.append(CallEdge(
                    caller=qualified_name,
                    callee=callee_qualified_name,
                    file_path=entity.relative_file_path,
                    line_number=entity.start_line,
                    call_type="direct",
                ))
    return edges


def _count_decision_points(body_text: str, language: str) -> int:
    """
    Approximate McCabe decision-point count via keyword search.

    A real implementation would walk the tree-sitter AST per function
    (complexity_analyzer.ComplexityAnalyzer), but entity_extractor.py does not
    retain the AST node on FunctionEntity — only the extracted body text. This
    counts branch/loop keywords directly, which McCabe complexity is defined
    over (if/elif/else/for/while/except/case/&&/||), understating complexity
    only for control flow spread across nested lambdas/comprehensions.
    """
    keywords = ["if", "elif", "else if", "for", "while", "except", "case", "catch"]
    count = sum(len(re.findall(rf"\b{kw}\b", body_text)) for kw in keywords)
    count += body_text.count("&&") + body_text.count("||")
    count += len(re.findall(r"\band\b|\bor\b", body_text)) if language == "python" else 0
    return count


def _raw_type_for(entity: FunctionEntity) -> str:
    """
    Maps a FunctionEntity back to the tree-sitter node type string that
    MetadataBuilder._resolve_chunk_type() expects. entity_extractor.py does
    not retain the original node type, so this reconstructs the closest
    equivalent from the fields it does keep (language, is_async).
    """
    if entity.language == "python":
        return "async_function_definition" if entity.is_async else "function_definition"
    if entity.language in ("javascript", "typescript"):
        return "method_definition" if entity.is_method else "function_declaration"
    if entity.language == "java":
        return "method_declaration"
    if entity.language == "go":
        return "method_declaration" if entity.is_method else "function_declaration"
    return "function_definition"


# ─────────────────────────────────────────────────────────────────────────────
# Main Task
# ─────────────────────────────────────────────────────────────────────────────

@celery_app.task(bind=True, name="ingest_repository")
def ingest_repository_task(
    self,
    repo_url: str,
    owner: str,
    repo: str,
    branch: Optional[str] = None,
    force_reindex: bool = False,
) -> dict:
    """
    Full ingestion pipeline for one repository. See Architecture-notes.md §6.

    Progress is reported via self.update_state(state="STARTED", meta={...})
    so GET /api/v1/ingest/status/{task_id} can show live progress.
    """
    t0 = time.perf_counter()
    redis = redis_lib.from_url(settings.redis_url)

    def progress(**meta):
        self.update_state(state="STARTED", meta=meta)

    try:
        # ── 1. Clone ─────────────────────────────────────────────────────────
        progress(stage="cloning", files_processed=0, files_total=0)
        clone_dir = str(Path(settings.repo_clone_dir) / f"{owner}_{repo}")
        repo_root = clone_repository(repo_url, target_dir=clone_dir, branch=branch)

        try:
            repo_meta = get_repo_metadata(repo_url)
        except Exception as e:
            logger.warning(f"Could not fetch GitHub metadata for {owner}/{repo}: {e}")
            repo_meta = {}

        # ── 2. Walk + filter ─────────────────────────────────────────────────
        source_files: list[SourceFile] = walk_repository(repo_root)

        if len(source_files) > settings.ingestion_max_files:
            raise RuntimeError(
                f"Repository has {len(source_files)} files, exceeding the hard "
                f"limit of {settings.ingestion_max_files}. Sharding is not "
                f"implemented in v1 (Architecture-notes.md §4)."
            )
        if len(source_files) > settings.ingestion_warn_threshold:
            logger.warning(
                f"{owner}/{repo} has {len(source_files)} files — above the "
                f"{settings.ingestion_warn_threshold} warn threshold."
            )

        # ── 3. Change detection (re-ingest short-circuit) ───────────────────
        # Re-ingesting is a full rebuild, not a per-file patch: the call graph,
        # node2vec structural vectors and in/out-degree metadata all depend on
        # the WHOLE repo, so re-processing only the changed files would
        # overwrite them with a partial view. What we can skip cheaply is the
        # case where nothing changed at all since the last successful index.
        collection_name = settings.qdrant_collection_name(owner, repo)
        detector = ChangeDetector(repo_path=repo_root)
        head_commit = detector._get_head_sha()
        already_indexed = redis.exists(f"indexed_repo:{owner}:{repo}")
        if (
            not force_reindex
            and already_indexed
            and detector.load_state().last_indexed_commit == head_commit
        ):
            logger.info(f"{owner}/{repo} is already indexed at {head_commit[:8]} — nothing to do.")
            redis.delete(f"ingest_active:{owner}:{repo}:{branch or 'default'}")
            return {
                "files_processed": 0,
                "chunks_indexed": 0,
                "functions_indexed": 0,
                "collection_name": collection_name,
                "duration_seconds": round(time.perf_counter() - t0, 2),
                "up_to_date": True,
            }

        files_total = len(source_files)
        progress(stage="parsing", files_processed=0, files_total=files_total)

        # ── 4. Parse ─────────────────────────────────────────────────────────
        parsed_files = parse_files_batch(source_files)

        # ── 5. Extract entities ──────────────────────────────────────────────
        file_entities: list[FileEntities] = extract_entities_batch(parsed_files)

        # ── 6. Build call graph ──────────────────────────────────────────────
        # Node id = "{relative_file_path}::{fully_qualified_name}" — e.g.
        # "app/auth.py::login" or "app/auth.py::Session.close" — matching
        # CodeChunk.fully_qualified_name so that enrich_with_call_graph() and
        # the structural-vector lookup below find every node, and so that two
        # same-named methods of different classes in one file stay distinct.
        builder = CallGraphBuilder(repo_name=f"{owner}/{repo}")
        all_functions: dict[str, tuple[FunctionEntity, str]] = {}

        for fe in file_entities:
            for fn in fe.functions:
                qualified = f"{fn.class_name}.{fn.name}" if fn.class_name else fn.name
                node_id = f"{fn.relative_file_path}::{qualified}"
                all_functions[node_id] = (fn, node_id)
                builder.add_function(FunctionNode(
                    qualified_name=node_id,
                    file_path=fn.relative_file_path,
                    start_line=fn.start_line,
                    end_line=fn.end_line,
                    language=fn.language,
                    is_public=not fn.name.startswith("_"),
                ))

        edges = _resolve_call_edges(all_functions)
        builder.add_calls_bulk(edges)
        call_graph = builder.graph
        logger.info(
            f"Call graph for {owner}/{repo}: {call_graph.number_of_nodes()} nodes, "
            f"{call_graph.number_of_edges()} edges ({len(edges)} resolved calls)."
        )

        # Persist the graph where retrieval/graph_retriever.py expects it.
        graph_dir = Path(settings.repo_clone_dir) / collection_name
        graph_dir.mkdir(parents=True, exist_ok=True)
        # Write-then-rename so the API never reads a half-written pickle.
        tmp_path = graph_dir / "call_graph.pkl.tmp"
        with open(tmp_path, "wb") as f:
            pickle.dump(call_graph, f)
        tmp_path.replace(graph_dir / "call_graph.pkl")

        # ── 7. Build chunks (metadata) ────────────────────────────────────────
        progress(stage="building_chunks", files_processed=0, files_total=files_total)
        meta_builder = MetadataBuilder(
            repo_url=repo_url, repo_owner=owner, repo_name=repo, commit_hash=head_commit,
        )

        # Maps each file's relative path to its raw source bytes, so we can
        # build a MODULE-level chunk per file below (see KNOWN GAPS).
        # extract_entities_batch() silently drops any parsed_files entry that
        # failed extraction, so file_entities and parsed_files are not
        # guaranteed to be aligned by index — look up by path instead.
        source_by_path: dict[str, bytes] = {
            pf.source_file.relative_path: pf.source_bytes for pf in parsed_files
        }

        all_chunks: list[CodeChunk] = []
        for i, fe in enumerate(file_entities):
            language_enum = Language(fe.language) if fe.language in Language._value2member_map_ else None
            if language_enum is None:
                continue

            raw_entities = []
            for fn in fe.functions:
                raw_entities.append({
                    "type": _raw_type_for(fn),
                    "name": fn.name,
                    "start_row": fn.start_line - 1,
                    "end_row": fn.end_line - 1,
                    "source_code": fn.body_text,
                    "signature": fn.signature,
                    "docstring": fn.docstring,
                    "decision_points": _count_decision_points(fn.body_text, fn.language),
                    "class_name": fn.class_name,
                    "imports": [imp.module_path for imp in fe.imports],
                })
            for cls in fe.classes:
                raw_entities.append({
                    "type": "class_definition" if fe.language == "python" else "class_declaration",
                    "name": cls.name,
                    "start_row": cls.start_line - 1,
                    "end_row": cls.end_line - 1,
                    "source_code": cls.body_text,
                    "signature": None,
                    "docstring": cls.docstring,
                    "decision_points": 0,
                    "class_name": None,
                    "imports": [imp.module_path for imp in fe.imports],
                })

            chunks = meta_builder.build_chunks_for_file(
                raw_entities=raw_entities, file_path=fe.source_file_path, language=language_enum,
            )
            all_chunks.extend(chunks)

            # ── Module-level chunk (KNOWN GAPS) ─────────────────────────────
            # build_chunks_for_file() above only covers functions and classes
            # extracted by entity_extractor.py — top-level statements outside
            # any function/class (module-level config, DB client/connection
            # setup, app initialization, etc.) are never visited by any
            # language's _extract_*_entities() and so never become a chunk.
            # This is a real, observed gap: a Flask/FastAPI app's DB
            # connection setup (`client = MongoClient(...)`) commonly lives
            # at module scope, and a query like "where is the data stored?"
            # would retrieve nothing that actually shows the connection
            # string/database — only the functions that *use* the already-
            # initialized `collection` variable, forcing the LLM to guess.
            # MetadataBuilder.build_module_chunk() already exists to cover
            # exactly this (whole-file content, embeddable_text built from
            # docstring+imports) but was never called anywhere in this
            # pipeline until now.
            source_bytes = source_by_path.get(fe.source_file_path)
            if source_bytes is not None:
                try:
                    module_chunk = meta_builder.build_module_chunk(
                        file_path=fe.source_file_path,
                        language=language_enum,
                        source_code=source_bytes.decode("utf-8", errors="replace"),
                        imports=[imp.module_path for imp in fe.imports],
                        docstring=fe.module_docstring,
                    )
                    all_chunks.append(module_chunk)
                except Exception as e:
                    logger.warning(
                        f"Skipping module chunk for '{fe.source_file_path}': {e}"
                    )

            if i % 25 == 0 or i == len(file_entities) - 1:
                progress(stage="building_chunks", files_processed=i + 1, files_total=files_total)

        all_chunks = meta_builder.enrich_with_call_graph(all_chunks, call_graph)

        # ── 8. Embed ─────────────────────────────────────────────────────────
        progress(stage="embedding_semantic", files_processed=0, files_total=len(all_chunks))
        code_embedder = CodeEmbedder()
        batch_result = code_embedder.embed_batch(
            [(chunk.embeddable_text, chunk.chunk_id) for chunk in all_chunks]
        )
        embeddings_by_id = {r.chunk_id: r.embedding for r in batch_result.results}
        for chunk in all_chunks:
            chunk.semantic_vector = embeddings_by_id.get(chunk.chunk_id)

        progress(stage="embedding_structural", files_processed=0, files_total=len(all_chunks))
        graph_embedder = GraphEmbedder()
        structural_vectors = graph_embedder.embed_graph(call_graph)
        for chunk in all_chunks:
            # Call graph nodes are keyed "{file_path}::{fully_qualified_name}" —
            # the same node_id convention used by retrieval/graph_retriever.py
            # and metadata_builder.enrich_with_call_graph (see module docstring).
            node_id = f"{chunk.file_path}::{chunk.fully_qualified_name}"
            structural = graph_embedder.get_node_vector(node_id, structural_vectors)
            chunk.structural_vector = structural.tolist()

        # ── 9. Index into Qdrant ─────────────────────────────────────────────
        progress(stage="indexing", files_processed=0, files_total=len(all_chunks))
        # Drop the previous index first so chunks for deleted/renamed
        # functions and files don't linger — this is a full rebuild (see
        # step 3). Done only now, after all the slow parse/embed work has
        # succeeded, to keep the window where the repo has no index short.
        qdrant = QdrantIndexer()
        if qdrant.collection_info(owner, repo).get("status") != "not_found":
            qdrant.delete_collection(owner, repo)
        qdrant.ensure_collection(owner, repo)
        qdrant.upsert_chunks(owner, repo, all_chunks)

        # ── 10. Persist index state + Redis "indexed" record ────────────────
        detector.save_state(
            indexed_files=detector.get_all_indexed_files(),
            indexed_file_count=files_total,
        )

        duration = time.perf_counter() - t0
        # Field set here must match api/routes/ingest.py's RepositoryInfo exactly —
        # list_repositories() does `RepositoryInfo(**json.loads(redis.get(key)))`.
        import datetime as _datetime
        languages_detected = sorted({fe.language for fe in file_entities})
        record = {
            "owner": owner,
            "repo": repo,
            "repo_url": repo_url,
            "collection_name": collection_name,
            "total_chunks": len(all_chunks),
            "indexed_at": _datetime.datetime.utcnow().isoformat(),
            "branch": branch or repo_meta.get("default_branch") or "default",
            "languages": languages_detected,
        }
        redis.set(f"indexed_repo:{owner}:{repo}", json.dumps(record))
        redis.delete(f"ingest_active:{owner}:{repo}:{branch or 'default'}")

        logger.success(
            f"Ingestion complete: {owner}/{repo} — {len(all_chunks)} chunks in {duration:.1f}s"
        )

        return {
            "files_processed": files_total,
            "chunks_indexed": len(all_chunks),
            "functions_indexed": len(all_functions),
            "collection_name": collection_name,
            "duration_seconds": round(duration, 2),
        }

    except Exception as exc:
        logger.exception(f"Ingestion failed for {owner}/{repo}: {exc}")
        redis.delete(f"ingest_active:{owner}:{repo}:{branch or 'default'}")
        raise
