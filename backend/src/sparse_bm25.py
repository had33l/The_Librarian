"""Memory-light BM25 (Okapi) with the same scoring as rank_bm25.BM25Okapi.

rank_bm25 keeps one Python dict per document, which costs hundreds of MB for
~17k chunks. This version stores an inverted index in a few numpy arrays
(tens of MB) and produces identical scores (k1=1.5, b=0.75, epsilon=0.25).
"""

import json
import re
from array import array
from collections import Counter

import numpy as np


def tokenize(text):
    """Split text into lowercase words for BM25 (same rule used for the evaluation)."""
    return re.findall(r"\b\w+\b", text.lower())


class SparseBM25:
    def __init__(self, tokenized_corpus, k1=1.5, b=0.75, epsilon=0.25):
        self.k1, self.b, self.epsilon = k1, b, epsilon
        self.corpus_size = 0
        vocab = {}
        term_ids, doc_ids, freqs = array("i"), array("i"), array("f")
        doc_len = array("i")

        for doc_id, tokens in enumerate(tokenized_corpus):
            self.corpus_size += 1
            doc_len.append(len(tokens))
            for term, freq in Counter(tokens).items():
                term_id = vocab.setdefault(term, len(vocab))
                term_ids.append(term_id)
                doc_ids.append(doc_id)
                freqs.append(freq)

        self.vocab = vocab
        self.doc_len = np.frombuffer(doc_len, dtype=np.int32).astype(np.float64)
        self.avgdl = float(self.doc_len.mean())

        term_ids = np.frombuffer(term_ids, dtype=np.int32)
        doc_ids = np.frombuffer(doc_ids, dtype=np.int32)
        freqs = np.frombuffer(freqs, dtype=np.float32).astype(np.float64)

        # Per (term, doc) BM25 weight without the idf factor.
        norm = self.k1 * (1 - self.b + self.b * self.doc_len[doc_ids] / self.avgdl)
        weights = freqs * (self.k1 + 1) / (freqs + norm)

        # Inverted index sorted by term (CSC layout).
        order = np.argsort(term_ids, kind="stable")
        self.postings_docs = doc_ids[order]
        self.postings_weights = weights[order]
        counts = np.bincount(term_ids, minlength=len(vocab))
        self.indptr = np.concatenate(([0], np.cumsum(counts)))

        del term_ids, doc_ids, freqs, norm, weights, order  # release temporaries

        # idf exactly as rank_bm25: log((N - n + .5) / (n + .5)), negatives
        # replaced by epsilon * average idf.
        n = counts.astype(np.float64)
        idf = np.log(self.corpus_size - n + 0.5) - np.log(n + 0.5)
        average_idf = idf.sum() / len(idf)
        idf[idf < 0] = self.epsilon * average_idf
        self.idf = idf

    def get_scores(self, query_tokens):
        scores = np.zeros(self.corpus_size, dtype=np.float64)
        for token in query_tokens:  # repeated query tokens count repeatedly
            term_id = self.vocab.get(token)
            if term_id is None:
                continue
            start, end = self.indptr[term_id], self.indptr[term_id + 1]
            scores[self.postings_docs[start:end]] += (
                self.idf[term_id] * self.postings_weights[start:end]
            )
        return scores

    # ---------- persistence (skips the expensive build at startup) ----------
    def save(self, path):
        terms = sorted(self.vocab, key=self.vocab.get)  # index == term id
        np.savez_compressed(
            path,
            terms=np.array(json.dumps(terms)),
            doc_len=self.doc_len,
            postings_docs=self.postings_docs,
            postings_weights=self.postings_weights,
            indptr=self.indptr,
            idf=self.idf,
        )

    @classmethod
    def load(cls, path):
        self = cls.__new__(cls)
        data = np.load(path, allow_pickle=False)
        terms = json.loads(str(data["terms"]))
        self.vocab = {term: i for i, term in enumerate(terms)}
        self.doc_len = data["doc_len"]
        self.corpus_size = len(self.doc_len)
        self.postings_docs = data["postings_docs"]
        self.postings_weights = data["postings_weights"]
        self.indptr = data["indptr"]
        self.idf = data["idf"]
        return self


if __name__ == "__main__":
    # Build the index shipped with the app:  python -m src.sparse_bm25
    from pathlib import Path

    base = Path(__file__).resolve().parents[1]
    with (base / "data" / "processed" / "chunks.json").open(encoding="utf-8") as file:
        chunks = json.load(file)
    index = SparseBM25(tokenize(chunk["text"]) for chunk in chunks)
    out = base / "indexes" / "bm25_index.npz"
    index.save(out)
    print(f"Saved BM25 index for {len(chunks)} chunks to {out}")
