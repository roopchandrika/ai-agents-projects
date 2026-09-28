"""Retrieval modes for the HotpotQA comparison. Every mode returns the same number of paragraphs (FINAL_K), so
differences come from *which* paragraphs are chosen, not how many.

vector         top FINAL_K by embedding similarity (the plain RAG baseline)
vector_rerank  top VECTOR_K by embedding, reranked by a cross-encoder: the graph's real competitor, since it
               isolates what reranking alone buys (bge-reranker-base: 80.0% -> 84.7% both-gold recall; the
               MS MARCO MiniLM models made it worse).
graph          graph expansion: entities named in the question and entities in the top vector hits are expanded
               up to two hops, and the paragraphs that mention the reached entities become candidates. This
               targets bridge questions, whose second paragraph is about an entity the question never names.
               GRAPH_STRATEGY picks how candidates become the final FINAL_K:
                 "rerank": vector top VECTOR_K + graph candidates, reranked together (default)
                 "slots":  top FINAL_K - GRAPH_SLOTS vector hits, then the GRAPH_SLOTS graph candidates most
                           similar to the question (no reranker needed)
               Relation triples between the chosen paragraphs go to the model too.
"""
import os
from dataclasses import dataclass, field
from functools import lru_cache

from sentence_transformers import CrossEncoder

from rag import config

from . import vectors
from .corpus import paragraphs

GRAPH_STRATEGY = os.getenv("GRAPH_STRATEGY", "rerank")
FINAL_K = 5
VECTOR_K = 20
GRAPH_SLOTS = 2
SEED_CHUNKS = 2       # entities in the top-N vector hits also seed the expansion (finds the bridge entity)
EXPAND_LIMIT = 30


@dataclass
class Retrieved:
    titles: list[str]
    triples: list[tuple[str, str, str]] = field(default_factory=list)
    linked_entities: list[str] = field(default_factory=list)
    graph_candidates: int = 0


@lru_cache(maxsize=1)
def reranker() -> CrossEncoder:
    return CrossEncoder(config.RERANK_MODEL)


def rerank(question: str, titles: list[str], k: int) -> list[str]:
    paras = paragraphs()
    scores = reranker().predict([(question, f"{t}. {paras[t].text}") for t in titles])
    return [t for _, t in sorted(zip(scores, titles), key=lambda x: -x[0])][:k]


def retrieve(question: str, mode: str, graph=None, linker=None) -> Retrieved:
    if mode == "vector":
        return Retrieved([t for t, _ in vectors.search(question, FINAL_K)])
    candidates = [t for t, _ in vectors.search(question, VECTOR_K)]
    if mode == "vector_rerank":
        return Retrieved(rerank(question, candidates, FINAL_K))
    if mode != "graph":
        raise ValueError(mode)

    linked = linker.link(question)
    seeds = list(dict.fromkeys(linked + graph.chunk_entities(candidates[:SEED_CHUNKS])))
    expanded = [t for t, _ in graph.expand(seeds, limit=EXPAND_LIMIT)] if seeds else []
    expanded = [t for t in dict.fromkeys(expanded) if t in paragraphs() and t not in candidates]
    if GRAPH_STRATEGY == "rerank":
        chosen = rerank(question, candidates + expanded, FINAL_K)
    else:
        base = candidates[:FINAL_K - GRAPH_SLOTS]
        sims = vectors.similarity(question, expanded)
        graph_picks = sorted(expanded, key=lambda t: -sims.get(t, 0.0))[:GRAPH_SLOTS]
        chosen = (base + graph_picks + candidates[FINAL_K - GRAPH_SLOTS:])[:FINAL_K]
    return Retrieved(chosen, graph.triples(chosen), linked, len(expanded))
