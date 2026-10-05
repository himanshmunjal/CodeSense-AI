"""
evaluation/latency_benchmark.py
════════════════════════════════════════════════════════════════════════════

WHY THIS FILE EXISTS
─────────────────────
query.py's docstring states the project's latency targets explicitly:
    p50: < 2 seconds
    p90: < 3 seconds
    p99: < 5 seconds
and the README's "Evaluation Results" table reports p50/p90/p99 of
1.1s / 2.4s / 4.8s on FastAPI with 342 files indexed, calling out
"Target was p90 < 3s. [checkmark]". This script is what measures that: it
sends real HTTP requests to POST /api/v1/query/ for an already-ingested
repo, times each one wall-clock, and computes percentiles.

WHY WALL-CLOCK CLIENT-SIDE TIMING (NOT THE SERVER'S latency_ms FIELD):
The response already carries a server-side `latency_ms` (see
QueryResponse.latency_ms in api/routes/query.py), but that number
excludes network round-trip time by design (its docstring says so
explicitly). This script measures BOTH: client-side wall-clock latency
(what a real user actually experiences, including network) and reports
the server-side latency_ms alongside it, so a large gap between the two
signals a network problem rather than a backend problem.

WHY SEQUENTIAL, NOT CONCURRENT, BY DEFAULT:
Percentile latency numbers are only meaningful if they reflect the
single-query experience: if we fire N requests concurrently, later
ones queue up and their latency reflects contention, not the per-query
cost the README table is describing. `--concurrency` is exposed for
anyone who explicitly wants a load-test-style run, but the default (1)
matches how the README's numbers were actually gathered.

USAGE
──────
    python latency_benchmark.py --repo-url https://github.com/tiangolo/fastapi
    python latency_benchmark.py --repo-url <url> --n 100 --concurrency 4
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx
from loguru import logger

THIS_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_PATH = THIS_DIR / "eval_dataset.json"
DEFAULT_RESULTS_DIR = THIS_DIR / "results"

BASE_URL = os.environ.get("CODESENSE_API_BASE_URL", "http://localhost:8000")
QUERY_ENDPOINT = f"{BASE_URL}/api/v1/query/"

# Per README / Architecture-notes.md / query.py docstring latency targets.
P50_TARGET_S = 2.0
P90_TARGET_S = 3.0
P99_TARGET_S = 5.0

REQUEST_TIMEOUT_S = 30.0


@dataclass
class LatencySample:
    query: str
    repo_url: str
    wall_clock_ms: float
    server_latency_ms: Optional[float]
    confidence: Optional[float]
    status_code: Optional[int]
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


def _load_queries_for_repo(dataset_path: Path, repo_url: str, n: Optional[int]) -> list[str]:
    """
    Pulls query strings from eval_dataset.json for the given repo, so the
    benchmark exercises realistic questions rather than a single canned
    query repeated N times (which would flatter caching behavior).

    If n exceeds the number of dataset entries for this repo, queries are
    cycled to reach the requested count — still realistic, just repeated.
    """
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Eval dataset not found at {dataset_path}. Pass --queries-file "
            f"or --dataset to point at a valid eval_dataset.json, or "
            f"use --query to supply a single ad-hoc query."
        )
    with open(dataset_path) as f:
        data = json.load(f)

    matching = [
        e["query"] for e in data.get("queries", [])
        if repo_url.lower() in e["repo_url"].lower()
    ]
    if not matching:
        raise ValueError(
            f"No eval_dataset.json entries match repo_url substring "
            f"'{repo_url}'. Available repos: "
            f"{sorted({e['repo_url'] for e in data.get('queries', [])})}"
        )

    if n is None:
        return matching

    # Cycle through if more samples requested than available queries.
    out = []
    i = 0
    while len(out) < n:
        out.append(matching[i % len(matching)])
        i += 1
    return out


def _check_backend_reachable(client: httpx.Client) -> None:
    """Same fail-fast pattern as run_eval.py — clear error, no cryptic traceback."""
    try:
        client.get(BASE_URL, timeout=5.0)
    except httpx.ConnectError as e:
        raise RuntimeError(
            f"Could not connect to the CodeSense backend at {BASE_URL}.\n"
            f"Is it running? Start it with:\n"
            f"    cd backend && uvicorn api.main:app --reload --port 8000\n"
            f"(Original error: {e})"
        ) from e
    except httpx.HTTPError:
        pass


def _run_one(client: httpx.Client, repo_url: str, query: str, max_results: int) -> LatencySample:
    """
    Sends a single query and records wall-clock latency around the HTTP
    call itself (connection + request + response, not including any local
    setup). This is the number a real client of the API would observe.
    """
    payload = {"repo_url": repo_url, "question": query, "max_results": max_results}
    t0 = time.perf_counter()
    try:
        resp = client.post(QUERY_ENDPOINT, json=payload, timeout=REQUEST_TIMEOUT_S)
    except httpx.HTTPError as e:
        wall_clock_ms = (time.perf_counter() - t0) * 1000
        return LatencySample(
            query=query, repo_url=repo_url, wall_clock_ms=wall_clock_ms,
            server_latency_ms=None, confidence=None, status_code=None,
            error=f"request failed: {e}",
        )
    wall_clock_ms = (time.perf_counter() - t0) * 1000

    if resp.status_code == 404:
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text
        return LatencySample(
            query=query, repo_url=repo_url, wall_clock_ms=wall_clock_ms,
            server_latency_ms=None, confidence=None, status_code=404,
            error=f"repo not ingested: {detail}",
        )

    if resp.status_code != 200:
        return LatencySample(
            query=query, repo_url=repo_url, wall_clock_ms=wall_clock_ms,
            server_latency_ms=None, confidence=None, status_code=resp.status_code,
            error=f"HTTP {resp.status_code}: {resp.text[:200]}",
        )

    try:
        body = resp.json()
    except Exception as e:
        return LatencySample(
            query=query, repo_url=repo_url, wall_clock_ms=wall_clock_ms,
            server_latency_ms=None, confidence=None, status_code=200,
            error=f"could not decode JSON: {e}",
        )

    return LatencySample(
        query=query, repo_url=repo_url, wall_clock_ms=wall_clock_ms,
        server_latency_ms=body.get("latency_ms"),
        confidence=body.get("confidence"),
        status_code=200,
    )


async def _run_all(
    repo_url: str, queries: list[str], max_results: int, concurrency: int
) -> list[LatencySample]:
    """
    Runs all queries either fully sequentially (concurrency=1, the default
    and the mode that produces meaningful single-query percentiles) or with
    bounded concurrency via a semaphore for anyone deliberately load-testing.
    """
    samples: list[LatencySample] = []
    semaphore = asyncio.Semaphore(concurrency)

    async def _worker(query: str) -> LatencySample:
        async with semaphore:
            # httpx.Client (sync) is run in a thread so we can bound
            # concurrency with a plain asyncio.Semaphore without needing
            # to manage a separate AsyncClient connection pool per worker.
            return await asyncio.to_thread(_run_one_with_client, repo_url, query, max_results)

    tasks = [_worker(q) for q in queries]
    for coro in asyncio.as_completed(tasks):
        samples.append(await coro)
    return samples


def _run_one_with_client(repo_url: str, query: str, max_results: int) -> LatencySample:
    with httpx.Client() as client:
        return _run_one(client, repo_url, query, max_results)


def _percentile(values: list[float], pct: float) -> float:
    """
    Nearest-rank percentile. We avoid numpy here since this file otherwise
    has zero third-party dependencies beyond httpx/loguru — no reason to
    pull in numpy just for a percentile calculation on a list of floats.
    """
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = max(0, min(len(sorted_vals) - 1, int(round(pct / 100 * (len(sorted_vals) - 1)))))
    return sorted_vals[k]


def compute_report(samples: list[LatencySample]) -> dict:
    """Aggregates raw samples into the percentile table the README shows."""
    ok = [s for s in samples if s.error is None]
    failed = [s for s in samples if s.error is not None]

    wall_clock = [s.wall_clock_ms / 1000.0 for s in ok]  # convert to seconds
    server_latency = [
        s.server_latency_ms / 1000.0 for s in ok if s.server_latency_ms is not None
    ]

    def _percentiles(values_s: list[float]) -> dict:
        if not values_s:
            return {"p50": None, "p90": None, "p99": None, "mean": None, "max": None}
        return {
            "p50": round(_percentile(values_s, 50), 3),
            "p90": round(_percentile(values_s, 90), 3),
            "p99": round(_percentile(values_s, 99), 3),
            "mean": round(statistics.mean(values_s), 3),
            "max": round(max(values_s), 3),
        }

    wall_clock_pcts = _percentiles(wall_clock)
    server_pcts = _percentiles(server_latency)

    p90 = wall_clock_pcts["p90"]
    p90_target_met = (p90 is not None) and (p90 < P90_TARGET_S)

    return {
        "n_total": len(samples),
        "n_ok": len(ok),
        "n_failed": len(failed),
        "wall_clock_seconds": wall_clock_pcts,
        "server_latency_seconds": server_pcts,
        "targets": {"p50": P50_TARGET_S, "p90": P90_TARGET_S, "p99": P99_TARGET_S},
        "p90_target_met": p90_target_met,
    }


def print_report(report: dict, samples: list[LatencySample]) -> None:
    wc = report["wall_clock_seconds"]
    sv = report["server_latency_seconds"]

    print("\n" + "=" * 78)
    print("CODESENSE LATENCY BENCHMARK")
    print("=" * 78)
    print(f"Queries run:   {report['n_total']}  (ok: {report['n_ok']}, failed: {report['n_failed']})")
    print()
    print(f"{'Percentile':<12}{'Wall-clock':>14}{'Server latency_ms':>20}")
    print("-" * 78)
    for label, key in (("p50", "p50"), ("p90", "p90"), ("p99", "p99")):
        wc_val = f"{wc[key]:.2f}s" if wc[key] is not None else "n/a"
        sv_val = f"{sv[key]:.2f}s" if sv[key] is not None else "n/a"
        print(f"{label:<12}{wc_val:>14}{sv_val:>20}")
    print("-" * 78)
    mean_wc = f"{wc['mean']:.2f}s" if wc["mean"] is not None else "n/a"
    max_wc = f"{wc['max']:.2f}s" if wc["max"] is not None else "n/a"
    print(f"{'mean':<12}{mean_wc:>14}")
    print(f"{'max':<12}{max_wc:>14}")
    print("=" * 78)

    targets = report["targets"]
    status = "PASS" if report["p90_target_met"] else "FAIL"
    p90_str = f"{wc['p90']:.2f}s" if wc["p90"] is not None else "n/a"
    print(
        f"\nTarget: p90 < {targets['p90']}s  →  measured p90 = {p90_str}  →  [{status}]"
    )
    if not report["p90_target_met"]:
        print(
            "WARNING: p90 latency target NOT met. Check Groq API latency, "
            "cross-encoder re-ranker load, or whether CodeBERT is warm "
            "(cold start adds ~3-5s per Architecture-notes.md)."
        )

    failed = [s for s in samples if s.error is not None]
    if failed:
        print(f"\n{len(failed)} quer{'y' if len(failed) == 1 else 'ies'} failed:")
        for s in failed[:10]:
            print(f"  [{s.status_code}] {s.query[:60]!r}: {s.error}")


def write_report(report: dict, samples: list[LatencySample], results_dir: Path) -> tuple[Path, Path]:
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    json_path = results_dir / f"latency_report_{timestamp}.json"
    md_path = results_dir / f"latency_report_{timestamp}.md"

    full = {
        "generated_at": timestamp,
        "base_url": BASE_URL,
        "report": report,
        "samples": [s.to_dict() for s in samples],
    }
    with open(json_path, "w") as f:
        json.dump(full, f, indent=2)

    wc = report["wall_clock_seconds"]
    status = "Met" if report["p90_target_met"] else "NOT met"
    md_lines = [
        f"# CodeSense Latency Benchmark ({timestamp})",
        "",
        f"Backend: `{BASE_URL}`",
        f"Queries run: {report['n_total']} (ok: {report['n_ok']}, failed: {report['n_failed']})",
        "",
        "| Percentile | Latency |",
        "|---|---|",
        f"| p50 | {wc['p50']}s |" if wc["p50"] is not None else "| p50 | n/a |",
        f"| p90 | {wc['p90']}s |" if wc["p90"] is not None else "| p90 | n/a |",
        f"| p99 | {wc['p99']}s |" if wc["p99"] is not None else "| p99 | n/a |",
        "",
        f"Target was p90 < {report['targets']['p90']}s. {status}.",
    ]
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines) + "\n")

    return json_path, md_path


async def _main_async(args: argparse.Namespace) -> int:
    with httpx.Client() as check_client:
        try:
            _check_backend_reachable(check_client)
        except RuntimeError as e:
            logger.error(str(e))
            return 1

    if args.query:
        queries = [args.query] * (args.n or 1)
    else:
        queries = _load_queries_for_repo(Path(args.dataset), args.repo_url, args.n)

    logger.info(
        f"Running {len(queries)} queries against {args.repo_url} "
        f"[concurrency={args.concurrency}] ..."
    )

    samples = await _run_all(args.repo_url, queries, args.max_results, args.concurrency)
    report = compute_report(samples)
    print_report(report, samples)

    results_dir = Path(args.results_dir)
    json_path, md_path = write_report(report, samples, results_dir)
    logger.info(f"Full JSON report written to {json_path}")
    logger.info(f"Markdown report written to {md_path}")

    if report["n_ok"] == 0:
        logger.error(
            "Every single query failed — the backend is reachable but no "
            "repo appears to be ingested for this repo_url. "
            "Run POST /api/v1/ingest first."
        )
        return 1

    return 0 if report["p90_target_met"] else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark POST /api/v1/query/ latency (p50/p90/p99) against an ingested repo."
    )
    parser.add_argument(
        "--repo-url", required=True,
        help="GitHub URL of an already-ingested repo, e.g. https://github.com/tiangolo/fastapi",
    )
    parser.add_argument(
        "--dataset", default=str(DEFAULT_DATASET_PATH),
        help="eval_dataset.json to pull realistic queries from (default: backend/evaluation/eval_dataset.json).",
    )
    parser.add_argument(
        "--n", type=int, default=None,
        help="Number of queries to run. Default: all dataset entries for --repo-url. "
             "Cycles through available queries if n exceeds the dataset size.",
    )
    parser.add_argument(
        "--query", default=None,
        help="Run a single ad-hoc query N times instead of pulling from eval_dataset.json.",
    )
    parser.add_argument(
        "--max-results", type=int, default=5,
        help="max_results per query request (default 5).",
    )
    parser.add_argument(
        "--concurrency", type=int, default=1,
        help="Concurrent in-flight requests (default 1 — sequential, matching how "
             "single-query percentile latency is meant to be measured).",
    )
    parser.add_argument(
        "--results-dir", default=str(DEFAULT_RESULTS_DIR),
        help="Directory to write JSON/Markdown reports to (default: backend/evaluation/results/)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exit_code = asyncio.run(_main_async(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
