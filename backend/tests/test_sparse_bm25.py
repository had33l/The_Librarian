"""The compact BM25 must score exactly like rank_bm25.BM25Okapi."""

import json

import numpy as np
import pytest
from rank_bm25 import BM25Okapi

import src.ask as ask
from src.sparse_bm25 import SparseBM25, tokenize

QUERIES = [
    "How do I split a string in Python?",
    "What does the with statement do",
    "difference between a list and a tuple",
    "asyncio gather exceptions",
    "zzzz unknownword python",
    "python python python decorator decorator",
]


@pytest.fixture(scope="module")
def chunks():
    with ask.CHUNKS_FILE.open(encoding="utf-8") as file:
        return json.load(file)


@pytest.fixture(scope="module")
def reference(chunks):
    return BM25Okapi([tokenize(chunk["text"]) for chunk in chunks])


@pytest.mark.parametrize("source", ["built", "shipped_file"])
def test_scores_match_rank_bm25(source, chunks, reference):
    if source == "built":
        index = SparseBM25(tokenize(chunk["text"]) for chunk in chunks)
    else:
        index = SparseBM25.load(ask.BM25_FILE)
    assert index.corpus_size == len(chunks)
    for query in QUERIES:
        expected = reference.get_scores(tokenize(query))
        actual = index.get_scores(tokenize(query))
        assert np.allclose(expected, actual, atol=1e-9)
        assert list(np.argsort(expected)[::-1][:5]) == list(np.argsort(actual)[::-1][:5])
