"""
tests/test_sources_used.py — inline-citation counting for the sources badge.

Regression coverage for "What calls full_dispatch_request?" against
pallets/flask: the answer cited 【src/flask/app.py:Flask.wsgi_app:1569-1619】
with full-width brackets, the ASCII-only pattern matched nothing, and the
badge showed "0/4 sources" on a correctly grounded answer.
"""

from types import SimpleNamespace

import pytest

from api.routes.query import _count_sources_used


def _sources(*paths: str) -> list:
    return [SimpleNamespace(file_path=p) for p in paths]


SOURCES = _sources("src/flask/app.py", "tests/test_basic.py", "examples/celery/app.py")


@pytest.mark.parametrize("citation", [
    "[src/flask/app.py:Flask.wsgi_app:L1569-L1619]",
    "[src/flask/app.py:Flask.wsgi_app:1569-1619]",
    "【src/flask/app.py:Flask.wsgi_app:1569-1619】",
    "`src/flask/app.py:Flask.wsgi_app:1569-1619`",
    "(src/flask/app.py:Flask.wsgi_app:1569–1619)",
    "[ src/flask/app.py:Flask.wsgi_app:L1569 - L1619 ]",
])
def test_citation_variants_counted(citation):
    answer = f"wsgi_app calls self.full_dispatch_request(ctx) {citation}."
    assert _count_sources_used(answer, SOURCES) == 1


def test_counts_each_cited_file_once():
    answer = (
        "See 【src/flask/app.py:Flask.wsgi_app:1569-1619】 and "
        "【src/flask/app.py:Flask.__call__:1621-1628】, plus "
        "[tests/test_basic.py:test_session:L291-L310]."
    )
    assert _count_sources_used(answer, SOURCES) == 2


def test_uncited_answer_counts_zero():
    answer = "Flask.wsgi_app calls full_dispatch_request in src/flask/app.py."
    assert _count_sources_used(answer, SOURCES) == 0
