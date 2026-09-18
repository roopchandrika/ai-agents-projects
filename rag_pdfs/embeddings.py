"""
embeddings.py
-------------
Isolates the embedding step so it can be swapped or faked in tests.

Default implementation: TF-IDF (scikit-learn), fit once over the corpus at
ingest time. This is a deliberate choice for this project -- it's fully
local (no model download, no API calls, no GPU), fast, and good enough to
demonstrate the retrieval pipeline end to end. It is NOT as good as a real
neural embedding model at capturing semantic similarity (e.g. it won't
match "car" to "automobile"), which is exactly the trade-off documented in
the README.

To upgrade to semantic embeddings, swap TfidfEmbedder for either:
  - sentence-transformers (local, e.g. 'all-MiniLM-L6-v2') -- see the
    commented-out class below, or
  - Voyage AI embeddings (Anthropic's recommended embeddings partner,
    https://docs.anthropic.com/en/docs/build-with-claude/embeddings)
without changing anything in ingest.py or retrieve.py, since both only
depend on the Embedder interface (`fit`, `embed`) below.
"""

from typing import Protocol

from sklearn.feature_extraction.text import TfidfVectorizer


class Embedder(Protocol):
    def fit(self, texts: list[str]) -> None: ...
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class TfidfEmbedder:
    """Local, dependency-light embedder. Must be fit on the corpus once
    (at ingest time) before embed() is called on new queries, since TF-IDF
    vectors are defined relative to a fixed vocabulary."""

    def __init__(self, max_features: int = 2000):
        self._vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
        self._fitted = False

    def fit(self, texts: list[str]) -> None:
        self._vectorizer.fit(texts)
        self._fitted = True

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not self._fitted:
            raise RuntimeError("TfidfEmbedder.fit() must be called before embed().")
        return self._vectorizer.transform(texts).toarray().tolist()


# --- Optional upgrade path (not used by default; requires extra deps) -------
#
# from sentence_transformers import SentenceTransformer
#
# class NeuralEmbedder:
#     def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
#         self._model = SentenceTransformer(model_name)
#
#     def fit(self, texts: list[str]) -> None:
#         pass  # neural embedders don't need corpus fitting
#
#     def embed(self, texts: list[str]) -> list[list[float]]:
#         return self._model.encode(texts).tolist()
