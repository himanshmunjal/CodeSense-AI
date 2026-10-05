"""
================================================================================
tests/test_ingestion_pipeline.py — Celery Ingestion Task, End to End
================================================================================

WHY THIS FILE EXISTS:
    tasks/celery_worker.py is where every pipeline stage is glued together,
    and glue is where naming mismatches hide: the call graph's node ids, the
    chunks' fully_qualified_name, and the structural-vector lookup all have to
    agree, or methods silently lose their graph signal.

    These tests run the REAL pipeline — git clone stand-in, tree-sitter
    parsing, entity extraction, call graph, chunk building — against a tiny
    throwaway git repository. Only infrastructure is mocked: Qdrant, Redis,
    and the two embedding models (no model downloads, no Docker needed).

WHAT IS VERIFIED:
    - A first ingest indexes functions, methods and module chunks.
    - Graph node ids match chunk names, so every function/method chunk gets
      its structural vector and in/out degree.
    - Two same-named methods in one file stay distinct (nodes AND points).
    - Re-ingesting an unchanged commit is a no-op.
    - A forced re-ingest rebuilds the collection from scratch.
================================================================================
"""

import pickle
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from config import settings
from tasks import celery_worker
from tasks.celery_worker import ingest_repository_task


SAMPLE_SOURCE = '''
class Reader:
    def close(self):
        return cleanup()


class Writer:
    def close(self):
        flush()
        return cleanup()


def cleanup():
    return True


def flush():
    return None
'''


class FakeRedis:
    """Just enough of redis-py for the worker: get/set/exists/delete."""

    def __init__(self):
        self.store = {}

    def set(self, key, value):
        self.store[key] = value

    def get(self, key):
        return self.store.get(key)

    def exists(self, key):
        return int(key in self.store)

    def delete(self, *keys):
        for key in keys:
            self.store.pop(key, None)


@pytest.fixture
def sample_repo(tmp_path):
    repo = tmp_path / "src_repo"
    repo.mkdir()
    (repo / "app.py").write_text(SAMPLE_SOURCE)
    for args in (
        ["init", "-q"],
        ["add", "."],
        ["-c", "user.email=t@example.com", "-c", "user.name=t", "commit", "-qm", "init"],
    ):
        subprocess.run(["git", *args], cwd=repo, check=True)
    return repo


@pytest.fixture
def pipeline(sample_repo, tmp_path, monkeypatch):
    """Patch infrastructure out of the worker; yield the mocks to assert on."""
    monkeypatch.setattr(settings, "repo_clone_dir", str(tmp_path / "clones"))

    redis = FakeRedis()
    qdrant = MagicMock()
    qdrant.collection_info.return_value = {"status": "not_found"}

    embedder = MagicMock()
    embedder.embed_batch.side_effect = lambda items: MagicMock(
        results=[MagicMock(chunk_id=cid, embedding=[0.0] * 384) for _, cid in items]
    )

    graph_embedder = MagicMock()
    graph_embedder.embed_graph.side_effect = lambda graph: {
        node: np.ones(128) for node in graph.nodes
    }
    graph_embedder.get_node_vector.side_effect = (
        lambda node_id, vectors: vectors.get(node_id, np.zeros(128))
    )

    with patch.object(celery_worker, "clone_repository", return_value=sample_repo), \
         patch.object(celery_worker, "get_repo_metadata", return_value={}), \
         patch.object(celery_worker.redis_lib, "from_url", return_value=redis), \
         patch.object(celery_worker, "QdrantIndexer", return_value=qdrant), \
         patch.object(celery_worker, "CodeEmbedder", return_value=embedder), \
         patch.object(celery_worker, "GraphEmbedder", return_value=graph_embedder), \
         patch.object(ingest_repository_task, "update_state"):
        yield MagicMock(redis=redis, qdrant=qdrant, embedder=embedder)


def _ingest(force_reindex=False):
    return ingest_repository_task.run(
        repo_url="https://github.com/test/repo",
        owner="test",
        repo="repo",
        force_reindex=force_reindex,
    )


def _upserted_chunks(qdrant):
    _, _, chunks = qdrant.upsert_chunks.call_args.args
    return chunks


class TestFirstIngest:
    def test_indexes_functions_methods_and_module(self, pipeline):
        result = _ingest()

        names = {c.fully_qualified_name for c in _upserted_chunks(pipeline.qdrant)}
        assert {"Reader.close", "Writer.close", "cleanup", "flush"} <= names
        assert result["functions_indexed"] == 4
        assert pipeline.redis.exists("indexed_repo:test:repo")

    def test_graph_nodes_match_chunk_names(self, pipeline):
        """
        Every function/method chunk must find its own node in the call graph —
        otherwise it gets a zero structural vector and no degree metadata.
        """
        _ingest()

        graph_path = Path(settings.repo_clone_dir) / "codesense_test_repo" / "call_graph.pkl"
        with open(graph_path, "rb") as f:
            graph = pickle.load(f)

        for chunk in _upserted_chunks(pipeline.qdrant):
            if chunk.chunk_type.value in ("function", "method"):
                node_id = f"{chunk.file_path}::{chunk.fully_qualified_name}"
                assert node_id in graph.nodes, node_id
                assert any(chunk.structural_vector), node_id

        assert graph.has_edge("app.py::Writer.close", "app.py::flush")
        assert graph.has_edge("app.py::Reader.close", "app.py::cleanup")
        assert not graph.has_edge("app.py::Reader.close", "app.py::flush")

    def test_same_named_methods_get_distinct_point_ids(self, pipeline):
        _ingest()

        close_chunks = [
            c for c in _upserted_chunks(pipeline.qdrant) if c.entity_name == "close"
        ]
        assert len(close_chunks) == 2
        assert len({c.chunk_id for c in close_chunks}) == 2


class TestReIngest:
    def test_unchanged_commit_is_a_no_op(self, pipeline):
        _ingest()
        pipeline.qdrant.reset_mock()

        result = _ingest()

        assert result["up_to_date"] is True
        pipeline.qdrant.upsert_chunks.assert_not_called()
        pipeline.qdrant.delete_collection.assert_not_called()

    def test_force_reindex_rebuilds_collection(self, pipeline):
        _ingest()
        pipeline.qdrant.reset_mock()
        pipeline.qdrant.collection_info.return_value = {"status": "green"}

        result = _ingest(force_reindex=True)

        pipeline.qdrant.delete_collection.assert_called_once_with("test", "repo")
        pipeline.qdrant.upsert_chunks.assert_called_once()
        assert result["chunks_indexed"] > 0
