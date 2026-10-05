"""
evaluation/run_eval.py
════════════════════════════════════════════════════════════════════════════

WHY THIS FILE EXISTS
─────────────────────
The README's "Hybrid Retrieval Experiment" and "Evaluation Results" sections
cite concrete numbers (84.7% Precision@5, 88.7% faithfulness, per-query-type
breakdowns) that were produced by running the 40-query eval set in
eval_dataset.json against a live, ingested repo. This script IS that harness
-- run it against a running backend with tiangolo/fastapi and/or
langchain-ai/langchain ingested and it reproduces (or refreshes) those
numbers.

WHAT IT MEASURES
─────────────────
1. Precision@5 (deterministic, no external dependency):
   For each eval entry, does ground_truth_file / ground_truth_function
   appear among the top-5 `sources` in the query response? This is a pure
   string-match metric against our own API's response shape -- it always
   works as long as the backend is reachable and the repo is ingested.

2. Confidence / refusal rate (also deterministic):
   Average `confidence` across all queries, and how many queries were
   refused because confidence fell below the grounding threshold
   (see Architecture-notes.md section 10 and query.py's
   retrieval_confidence_threshold).

3. RAGAS faithfulness + answer_relevancy (best-effort, needs network + an
   LLM judge):
   These require the `ragas` package and a working judge-model API call
   (RAGAS defaults to OpenAI; this project uses Groq elsewhere, so RAGAS
   here is wired to whatever OPENAI_API_KEY/RAGAS is configured with in the
   environment -- see _try_ragas_eval for exactly what's attempted). If
   ragas isn't importable, or the judge call fails (no network, no key,
   rate limited), we catch it and report those two metrics as `null` rather
   than crashing the whole run. Precision@5 and confidence are NOT
   RAGAS-dependent, so a run with zero network access still produces a
   usable report.

WHY BATCH-QUERY INSTEAD OF ONE-BY-ONE:
   The backend exposes POST /api/v1/query/batch specifically "used by
   run_eval.py" (see its docstring in api/routes/query.py). We use it here
   to cut wall-clock time roughly to that of the slowest single query in
   a repo's batch, matching the intent documented there. We still fall back
   to sequential single-query calls if batch is unavailable (404/405),
   since a partially-implemented backend shouldn't block the harness.

USAGE
──────
    # Against a locally running backend (default http://localhost:8000):
    python run_eval.py

    # Against a different backend / subset of the dataset:
    CODESENSE_API_BASE_URL=http://localhost:8000 python run_eval.py --repo fastapi
    python run_eval.py --dataset eval_dataset.json --max-results 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import httpx
from loguru import logger

# ── Constants ────────────────────────────────────────────────────────────────

THIS_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_PATH = THIS_DIR / "eval_dataset.json"
DEFAULT_RESULTS_DIR = THIS_DIR / "results"

# Mirrors the default retrieval_confidence_threshold in config.py (0.5,
# recalibrated for bge-small-en-v1.5). Duplicated here (not imported)
# because run_eval.py is meant to be runnable standalone against a remote
# backend without importing the backend's Python package at all — override
# with CODESENSE_CONFIDENCE_THRESHOLD if the backend uses a different value.
CONFIDENCE_THRESHOLD = float(os.environ.get("CODESENSE_CONFIDENCE_THRESHOLD", "0.5"))

BASE_URL = os.environ.get("CODESENSE_API_BASE_URL", "http://localhost:8000")
QUERY_ENDPOINT = f"{BASE_URL}/api/v1/query/"
BATCH_ENDPOINT = f"{BASE_URL}/api/v1/query/batch"

REQUEST_TIMEOUT_S = 60.0


# ── Result data structures ──────────────────────────────────────────────────

@dataclass
class QueryEvalResult:
    """
    Holds everything we know about one eval-set entry after querying the
    backend: what we asked, what came back, and the metrics we could
    derive from it.
    """
    id: str
    repo_url: str
    query: str
    query_type: str
    ground_truth_file: str
    ground_truth_function: str

    # Populated after the HTTP call, or left as error markers if it failed.
    answer: str = ""
    confidence: float = 0.0
    detected_query_type: str = ""
    sources: list[dict] = field(default_factory=list)
    latency_ms: float = 0.0
    error: Optional[str] = None

    # Derived metrics
    hit_at_5: bool = False
    refused: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


# ── Dataset loading ──────────────────────────────────────────────────────────

def load_dataset(path: Path) -> list[dict]:
    """
    Loads eval_dataset.json and returns the list of query entries.

    Raises a clear error (rather than a raw KeyError) if the file is
    missing or malformed, since this is the very first thing that runs
    and a confusing traceback here wastes the most time.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Eval dataset not found at {path}. "
            f"Expected backend/evaluation/eval_dataset.json."
        )
    with open(path) as f:
        data = json.load(f)

    queries = data.get("queries")
    if not queries:
        raise ValueError(
            f"{path} has no 'queries' array — nothing to evaluate."
        )
    return queries


# ── Backend interaction ──────────────────────────────────────────────────────

def _check_backend_reachable(client: httpx.Client) -> None:
    """
    Fails fast with an actionable message if the backend isn't running,
    instead of letting every single query fail with an opaque connection
    error. This is the single most common failure mode for anyone running
    this script without first starting `uvicorn api.main:app`.
    """
    try:
        # We don't rely on a specific health endpoint existing; a
        # connection-level failure is enough signal either way.
        client.get(BASE_URL, timeout=5.0)
    except httpx.ConnectError as e:
        raise RuntimeError(
            f"Could not connect to the CodeSense backend at {BASE_URL}.\n"
            f"Is it running? Start it with:\n"
            f"    cd backend && uvicorn api.main:app --reload --port 8000\n"
            f"Or point this script at a different host via "
            f"CODESENSE_API_BASE_URL.\n(Original error: {e})"
        ) from e
    except httpx.HTTPError:
        # Any HTTP-level response (even a 404) means the server IS up.
        pass


def _single_query(
    client: httpx.Client, entry: dict, max_results: int
) -> QueryEvalResult:
    """
    Runs one eval entry through POST /api/v1/query/ and maps the response
    onto a QueryEvalResult. Any failure (network, 404 repo-not-ingested,
    500, malformed JSON) is captured on `error` rather than raised, so one
    bad entry doesn't abort the whole eval run.
    """
    result = QueryEvalResult(
        id=entry["id"],
        repo_url=entry["repo_url"],
        query=entry["query"],
        query_type=entry["query_type"],
        ground_truth_file=entry["ground_truth_file"],
        ground_truth_function=entry["ground_truth_function"],
    )

    payload = {
        "repo_url": entry["repo_url"],
        "question": entry["query"],
        "max_results": max_results,
    }

    try:
        resp = client.post(QUERY_ENDPOINT, json=payload, timeout=REQUEST_TIMEOUT_S)
    except httpx.HTTPError as e:
        result.error = f"request failed: {e}"
        return result

    if resp.status_code == 404:
        # This is the _verify_repo_indexed() 404 from query.py — the most
        # actionable failure mode, so we surface its detail directly.
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text
        result.error = f"repo not ingested (404): {detail}"
        return result

    if resp.status_code != 200:
        result.error = f"HTTP {resp.status_code}: {resp.text[:300]}"
        return result

    try:
        body = resp.json()
    except Exception as e:
        result.error = f"could not decode JSON response: {e}"
        return result

    result.answer = body.get("answer", "")
    result.confidence = float(body.get("confidence", 0.0))
    result.detected_query_type = body.get("query_type", "unknown")
    result.sources = body.get("sources", [])
    result.latency_ms = float(body.get("latency_ms", 0.0))
    result.refused = result.confidence < CONFIDENCE_THRESHOLD
    result.hit_at_5 = _is_hit(result.sources, entry, k=5)
    return result


def _is_hit(sources: list[dict], entry: dict, k: int) -> bool:
    """
    Precision@5 check: does either the ground-truth file OR the ground-truth
    function name appear among the top-k sources?

    WHY "FILE OR FUNCTION" INSTEAD OF REQUIRING BOTH:
    Ground-truth function names in eval_dataset.json are sometimes a class
    (e.g. "ChatPromptTemplate") or a specific method within a larger file
    (e.g. "FastAPI.__init__"), while the chunker in this system may return
    the enclosing function/class under a slightly different label depending
    on how tree-sitter split it. Matching on file_path is the more robust
    signal for whether retrieval found "the right place in the codebase";
    matching function_name substrings gives partial credit for cases where
    the file matched but under a different chunk. We treat a hit as: the
    ground-truth file path matches AND (loosely) the function name appears
    as a substring of the returned function_name, OR an exact file match by
    itself for whole-file/summarization-style ground truths.
    """
    gt_file = entry["ground_truth_file"].lower()
    gt_func = entry["ground_truth_function"].lower()

    for source in sources[:k]:
        src_file = str(source.get("file_path", "")).lower()
        src_func = str(source.get("function_name", "")).lower()

        if gt_file not in src_file and src_file not in gt_file:
            continue

        # File matches. For lookup/relational/analytical queries with a
        # specific function target, also require a loose function-name
        # match. For summarization queries the file match alone is
        # sufficient (the ground truth function is just a representative
        # anchor for the whole file).
        if entry["query_type"] == "summarization":
            return True

        # Loose match: either name contains the other, or they share the
        # base identifier after stripping a "Class.method" qualifier.
        gt_func_base = gt_func.split(".")[-1]
        if gt_func in src_func or src_func in gt_func or gt_func_base in src_func:
            return True

    return False


async def _run_batch_for_repo(
    async_client: httpx.AsyncClient,
    repo_url: str,
    entries: list[dict],
    max_results: int,
) -> list[QueryEvalResult]:
    """
    Attempts POST /api/v1/query/batch for all entries belonging to one repo.
    Falls back to sequential single-query calls if the batch endpoint isn't
    available (older backend, 404/405) or returns an unexpected shape.

    NOTE: BatchQueryRequest caps at 20 questions (see api/routes/query.py),
    so for repos with more than 20 eval entries we chunk into groups of 20.
    """
    results: list[QueryEvalResult] = []
    CHUNK = 20

    for i in range(0, len(entries), CHUNK):
        chunk = entries[i : i + CHUNK]
        payload = {
            "repo_url": repo_url,
            "questions": [e["query"] for e in chunk],
        }
        try:
            resp = await async_client.post(
                BATCH_ENDPOINT, json=payload, timeout=REQUEST_TIMEOUT_S * len(chunk)
            )
        except httpx.HTTPError as e:
            logger.warning(f"Batch endpoint unreachable ({e}); falling back to sequential.")
            results.extend(_sequential_fallback(chunk, max_results))
            continue

        if resp.status_code == 404:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            for entry in chunk:
                r = QueryEvalResult(
                    id=entry["id"], repo_url=repo_url, query=entry["query"],
                    query_type=entry["query_type"],
                    ground_truth_file=entry["ground_truth_file"],
                    ground_truth_function=entry["ground_truth_function"],
                    error=f"repo not ingested (404): {detail}",
                )
                results.append(r)
            continue

        if resp.status_code != 200:
            logger.warning(
                f"Batch endpoint returned {resp.status_code}; falling back to sequential."
            )
            results.extend(_sequential_fallback(chunk, max_results))
            continue

        try:
            body = resp.json()
            batch_results = body["results"]
        except Exception as e:
            logger.warning(f"Malformed batch response ({e}); falling back to sequential.")
            results.extend(_sequential_fallback(chunk, max_results))
            continue

        for entry, res_body in zip(chunk, batch_results):
            result = QueryEvalResult(
                id=entry["id"], repo_url=repo_url, query=entry["query"],
                query_type=entry["query_type"],
                ground_truth_file=entry["ground_truth_file"],
                ground_truth_function=entry["ground_truth_function"],
                answer=res_body.get("answer", ""),
                confidence=float(res_body.get("confidence", 0.0)),
                detected_query_type=res_body.get("query_type", "unknown"),
                sources=res_body.get("sources", []),
                latency_ms=float(res_body.get("latency_ms", 0.0)),
            )
            result.refused = result.confidence < CONFIDENCE_THRESHOLD
            result.hit_at_5 = _is_hit(result.sources, entry, k=5)
            results.append(result)

    return results


def _sequential_fallback(entries: list[dict], max_results: int) -> list[QueryEvalResult]:
    """Runs entries one at a time via the single-query endpoint (sync httpx)."""
    out = []
    with httpx.Client() as client:
        for entry in entries:
            out.append(_single_query(client, entry, max_results))
    return out


# ── RAGAS (best-effort) ──────────────────────────────────────────────────────

def _try_ragas_eval(results: list[QueryEvalResult]) -> dict[str, Optional[float]]:
    """
    Attempts to compute RAGAS faithfulness + answer_relevancy over the
    successfully-answered (non-refused, non-error) queries.

    WHY THIS IS WRAPPED SO DEFENSIVELY:
    RAGAS needs: the `ragas` + `datasets` packages installed (they're in
    requirements.txt), AND a working LLM-judge API call, which typically
    needs network access and an API key (RAGAS's default judge is an
    OpenAI model). In this harness's sandboxed/dry-run context neither of
    those may be available, so every failure mode here is caught and
    reported as `null` metrics with a logged reason instead of crashing
    the whole evaluation run. Precision@5 and confidence do NOT depend on
    this function succeeding.

    Returns:
        {"faithfulness": float | None, "answer_relevancy": float | None,
         "ragas_error": str | None}
    """
    answerable = [
        r for r in results
        if not r.error and not r.refused and r.answer and r.sources
    ]

    if not answerable:
        return {
            "faithfulness": None,
            "answer_relevancy": None,
            "ragas_error": "No non-refused, non-error answers available to score.",
        }

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy
    except ImportError as e:
        logger.warning(f"RAGAS not available ({e}) — skipping faithfulness/relevancy.")
        return {
            "faithfulness": None,
            "answer_relevancy": None,
            "ragas_error": f"ragas/datasets not importable: {e}",
        }

    try:
        ragas_dataset = Dataset.from_dict({
            "question": [r.query for r in answerable],
            "answer": [r.answer for r in answerable],
            "contexts": [
                [s.get("snippet", "") for s in r.sources] or [""]
                for r in answerable
            ],
            # RAGAS's "ground_truth" column enables context_recall/answer
            # correctness too, but faithfulness + answer_relevancy alone
            # don't require it. We still don't have it wired from
            # ground_truth_answer_summary here to keep this call minimal
            # and closer to what generator.py actually produces.
        })

        # This is the step that needs network + a judge-model API key.
        # If GROQ_API_KEY/OPENAI_API_KEY is missing or unreachable, this
        # raises and we fall into the except block below.
        scores = evaluate(ragas_dataset, metrics=[faithfulness, answer_relevancy])
        scores_df = scores.to_pandas()

        return {
            "faithfulness": float(scores_df["faithfulness"].mean()),
            "answer_relevancy": float(scores_df["answer_relevancy"].mean()),
            "ragas_error": None,
        }
    except Exception as e:
        # Broad catch is intentional: RAGAS + its LLM client dependency
        # chain can raise many different exception types (auth errors,
        # rate limits, connection errors, missing event loop, etc.) and
        # we want ALL of them to degrade to "metric unavailable" rather
        # than crash run_eval.py.
        logger.warning(f"RAGAS evaluation failed — reporting null scores. Error: {e}")
        return {
            "faithfulness": None,
            "answer_relevancy": None,
            "ragas_error": str(e),
        }


# ── Aggregation ──────────────────────────────────────────────────────────────

def aggregate(results: list[QueryEvalResult]) -> dict:
    """
    Rolls up per-query results into the overall + per-query-type summary
    table shown in the README's "Evaluation Results" section.
    """
    def _agg(subset: list[QueryEvalResult]) -> dict:
        n = len(subset)
        answered = [r for r in subset if not r.error]
        if n == 0:
            return {
                "n": 0, "precision_at_5": None, "avg_confidence": None,
                "refusal_rate": None, "error_rate": None,
            }
        precision_at_5 = (
            sum(1 for r in answered if r.hit_at_5) / len(answered)
            if answered else None
        )
        avg_confidence = (
            statistics.mean(r.confidence for r in answered) if answered else None
        )
        refusal_rate = (
            sum(1 for r in answered if r.refused) / len(answered) if answered else None
        )
        return {
            "n": n,
            "precision_at_5": round(precision_at_5, 4) if precision_at_5 is not None else None,
            "avg_confidence": round(avg_confidence, 4) if avg_confidence is not None else None,
            "refusal_rate": round(refusal_rate, 4) if refusal_rate is not None else None,
            "error_rate": round((n - len(answered)) / n, 4),
        }

    query_types = sorted({r.query_type for r in results})
    by_type = {qt: _agg([r for r in results if r.query_type == qt]) for qt in query_types}

    ragas_scores = _try_ragas_eval(results)

    return {
        "overall": _agg(results),
        "by_query_type": by_type,
        "ragas": ragas_scores,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
    }


# ── Reporting ─────────────────────────────────────────────────────────────────

def _fmt_pct(x: Optional[float]) -> str:
    return f"{x * 100:.1f}%" if x is not None else "n/a"


def print_summary_table(summary: dict, results: list[QueryEvalResult]) -> None:
    """Prints a README-style summary table to stdout."""
    overall = summary["overall"]
    ragas = summary["ragas"]

    print("\n" + "=" * 78)
    print("CODESENSE EVALUATION SUMMARY")
    print("=" * 78)
    print(f"Total queries:        {overall['n']}")
    print(f"Precision@5:          {_fmt_pct(overall['precision_at_5'])}")
    print(f"Avg confidence:       {overall['avg_confidence'] if overall['avg_confidence'] is not None else 'n/a'}")
    print(f"Refusal rate (<{CONFIDENCE_THRESHOLD}): {_fmt_pct(overall['refusal_rate'])}")
    print(f"Error rate:           {_fmt_pct(overall['error_rate'])}")
    print(f"Faithfulness (RAGAS): {_fmt_pct(ragas['faithfulness'])}"
          + ("" if ragas["ragas_error"] is None else f"  [unavailable: {ragas['ragas_error']}]"))
    print(f"Answer relevancy:     {_fmt_pct(ragas['answer_relevancy'])}")

    print("\nBreakdown by query type:")
    print(f"{'Type':<15}{'n':>4}{'Precision@5':>14}{'Avg Confidence':>16}{'Refusal Rate':>14}")
    print("-" * 78)
    for qt, stats in summary["by_query_type"].items():
        avg_conf = stats["avg_confidence"]
        avg_conf_str = f"{avg_conf:.3f}" if avg_conf is not None else "n/a"
        print(
            f"{qt:<15}{stats['n']:>4}"
            f"{_fmt_pct(stats['precision_at_5']):>14}"
            f"{avg_conf_str:>16}"
            f"{_fmt_pct(stats['refusal_rate']):>14}"
        )
    print("=" * 78)

    errors = [r for r in results if r.error]
    if errors:
        print(f"\n{len(errors)} quer{'y' if len(errors) == 1 else 'ies'} failed to run:")
        for r in errors[:10]:
            print(f"  [{r.id}] {r.error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more (see full results JSON).")


def write_reports(summary: dict, results: list[QueryEvalResult], results_dir: Path) -> tuple[Path, Path]:
    """
    Writes both a machine-readable JSON report and a human-readable
    Markdown report (matching README table formatting) to
    backend/evaluation/results/.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    json_path = results_dir / f"eval_report_{timestamp}.json"
    md_path = results_dir / f"eval_report_{timestamp}.md"

    full_report = {
        "generated_at": timestamp,
        "base_url": BASE_URL,
        "summary": summary,
        "results": [r.to_dict() for r in results],
    }
    with open(json_path, "w") as f:
        json.dump(full_report, f, indent=2)

    md_lines = [
        f"# CodeSense Evaluation Report ({timestamp})",
        "",
        f"Backend: `{BASE_URL}`",
        "",
        "## Answer Quality",
        "",
        "| Metric | Score |",
        "|---|---|",
        f"| Precision@5 | {_fmt_pct(summary['overall']['precision_at_5'])} |",
        f"| Faithfulness (RAGAS) | {_fmt_pct(summary['ragas']['faithfulness'])} |",
        f"| Answer relevancy (RAGAS) | {_fmt_pct(summary['ragas']['answer_relevancy'])} |",
        f"| Avg confidence | {summary['overall']['avg_confidence']} |",
        f"| Refusal rate (< {CONFIDENCE_THRESHOLD}) | {_fmt_pct(summary['overall']['refusal_rate'])} |",
        "",
        "## Breakdown by Query Type",
        "",
        "| Query Type | n | Precision@5 | Avg Confidence | Refusal Rate |",
        "|---|---|---|---|---|",
    ]
    for qt, stats in summary["by_query_type"].items():
        avg_conf = f"{stats['avg_confidence']:.3f}" if stats["avg_confidence"] is not None else "n/a"
        md_lines.append(
            f"| {qt} | {stats['n']} | {_fmt_pct(stats['precision_at_5'])} | "
            f"{avg_conf} | {_fmt_pct(stats['refusal_rate'])} |"
        )

    if summary["ragas"]["ragas_error"]:
        md_lines += [
            "",
            f"> RAGAS metrics unavailable this run: {summary['ragas']['ragas_error']}",
        ]

    errors = [r for r in results if r.error]
    if errors:
        md_lines += ["", "## Failed Queries", ""]
        for r in errors:
            md_lines.append(f"- `{r.id}`: {r.error}")

    with open(md_path, "w") as f:
        f.write("\n".join(md_lines) + "\n")

    return json_path, md_path


# ── Entry point ───────────────────────────────────────────────────────────────

async def _main_async(args: argparse.Namespace) -> int:
    dataset_path = Path(args.dataset)
    entries = load_dataset(dataset_path)

    if args.repo:
        entries = [e for e in entries if args.repo.lower() in e["repo_url"].lower()]
        if not entries:
            logger.error(f"No dataset entries match --repo filter '{args.repo}'.")
            return 1

    logger.info(f"Loaded {len(entries)} eval queries from {dataset_path}")
    logger.info(f"Target backend: {BASE_URL}")

    with httpx.Client() as check_client:
        try:
            _check_backend_reachable(check_client)
        except RuntimeError as e:
            logger.error(str(e))
            return 1

    # Group by repo so each repo's batch call is independent — different
    # repos may or may not be ingested, and we want partial results even
    # if one repo fails entirely.
    by_repo: dict[str, list[dict]] = {}
    for e in entries:
        by_repo.setdefault(e["repo_url"], []).append(e)

    all_results: list[QueryEvalResult] = []
    async with httpx.AsyncClient() as async_client:
        for repo_url, repo_entries in by_repo.items():
            logger.info(f"Running {len(repo_entries)} queries against {repo_url} ...")
            repo_results = await _run_batch_for_repo(
                async_client, repo_url, repo_entries, args.max_results
            )
            all_results.extend(repo_results)

    summary = aggregate(all_results)
    print_summary_table(summary, all_results)

    results_dir = Path(args.results_dir)
    json_path, md_path = write_reports(summary, all_results, results_dir)
    logger.info(f"Full JSON report written to {json_path}")
    logger.info(f"Markdown report written to {md_path}")

    n_errors = sum(1 for r in all_results if r.error)
    if n_errors == len(all_results):
        logger.error(
            "Every single query failed — the backend is reachable but no "
            "repo appears to be ingested. Run POST /api/v1/ingest first."
        )
        return 1

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the CodeSense retrieval + generation evaluation suite."
    )
    parser.add_argument(
        "--dataset", default=str(DEFAULT_DATASET_PATH),
        help="Path to eval_dataset.json (default: backend/evaluation/eval_dataset.json)",
    )
    parser.add_argument(
        "--results-dir", default=str(DEFAULT_RESULTS_DIR),
        help="Directory to write JSON/Markdown reports to (default: backend/evaluation/results/)",
    )
    parser.add_argument(
        "--repo", default=None,
        help="Substring filter on repo_url, e.g. 'fastapi' to only run that repo's queries.",
    )
    parser.add_argument(
        "--max-results", type=int, default=5,
        help="max_results to request per query (default 5, matching Precision@5).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exit_code = asyncio.run(_main_async(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
