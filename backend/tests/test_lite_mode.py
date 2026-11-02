"""Lite mode (ONNX encoder + compact BM25) tests, using real chunks and embeddings.

fastembed is replaced by a tiny fake so the tests need no model download:
it returns the stored embedding for known chunk texts and a random unit vector
for anything else (e.g. a user question).
"""

import json
import sys
import types

import numpy as np
import pytest

import src.ask as ask
from src.onnx_encoder import OnnxEncoder


@pytest.fixture(scope="module")
def stored():
    with ask.CHUNKS_FILE.open(encoding="utf-8") as file:
        chunks = json.load(file)
    return chunks, np.load(ask.INDEX_FILE)


def install_fake_fastembed(monkeypatch, chunks, embeddings, corrupt=False):
    lookup = {chunk["text"]: embeddings[i] for i, chunk in enumerate(chunks)}
    rng = np.random.default_rng(1)

    class FakeTextEmbedding:
        def __init__(self, **kwargs):
            pass

        def embed(self, texts):
            for text in texts:
                vector = lookup.get(text)
                if vector is None or corrupt:
                    vector = rng.normal(size=384)
                yield vector * 3.0  # unnormalised on purpose: encoder must normalise

    module = types.ModuleType("fastembed")
    module.TextEmbedding = FakeTextEmbedding
    monkeypatch.setitem(sys.modules, "fastembed", module)


def test_onnx_encoder_returns_normalised_vectors(monkeypatch, stored):
    install_fake_fastembed(monkeypatch, *stored)
    vectors = OnnxEncoder().encode(["a question", "another"])
    assert vectors.shape == (2, 384)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


def test_lite_loader_and_retrieval_end_to_end(monkeypatch, stored):
    monkeypatch.setattr(ask, "LITE_MODE", True)
    install_fake_fastembed(monkeypatch, *stored)
    chunks, embeddings, bm25, encoder = ask.load_retriever()
    assert len(chunks) == len(embeddings) == bm25.corpus_size
    results, latency_ms = ask.retrieve(
        "How do I split a string in Python?", chunks, embeddings, bm25, encoder
    )
    assert len(results) == ask.TOP_K and latency_ms > 0
    assert all("url" in chunk for chunk in results)


def test_lite_loader_rejects_encoder_that_does_not_match_index(monkeypatch, stored):
    monkeypatch.setattr(ask, "LITE_MODE", True)
    install_fake_fastembed(monkeypatch, *stored, corrupt=True)
    with pytest.raises(ValueError, match="does not match"):
        ask.load_retriever()
