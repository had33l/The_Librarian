"""Query encoder that runs all-MiniLM-L6-v2 through ONNX Runtime (no PyTorch).

Same model weights as sentence-transformers, so it matches the stored
embeddings, but it needs roughly 150-250 MB of RAM instead of ~1 GB.
Exposes the small part of the SentenceTransformer API the app uses.
"""

import os

import numpy as np


class OnnxEncoder:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        from fastembed import TextEmbedding

        self._model = TextEmbedding(
            model_name=model_name,
            cache_dir=os.getenv("FASTEMBED_CACHE_PATH"),
            threads=1,
        )

    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True, **_):
        vectors = np.array(list(self._model.embed(list(texts))), dtype=np.float32)
        if normalize_embeddings:
            vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors
