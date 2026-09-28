# rag-platform

A plain RAG baseline over ~100 Wikipedia articles on space exploration, plus a repeatable eval harness.
Later projects (semantic caching and routing, multi-tenant ACLs, GraphRAG) are measured against this baseline.

**Pipeline:** Wikipedia articles → 450-token chunks with 75-token overlap → `bge-small-en-v1.5` embeddings →
Qdrant → top 5 chunks → Claude answers with numbered citations → FastAPI `/ask`.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env            # then put your ANTHROPIC_API_KEY in .env
```

## Build the index

```bash
python -m scripts.fetch_corpus    # downloads the articles to data/corpus/ (skips ones already there)
python -m rag.ingest              # chunks, embeds and indexes them (~2,000 chunks)
```

The corpus is committed on purpose: re-fetching Wikipedia later can change the text and break the eval set.

## Run the API

```bash
uvicorn rag.api:app --reload
```

```bash
curl -X POST localhost:8000/ask -H "Content-Type: application/json" -d "{\"question\": \"What did DART hit?\"}"
```

The response includes the answer, the numbered sources, tokens, cost and latency.

## Evaluate

1. **Draft questions:** `python -m eval.build_eval --n 80` samples chunks across documents and has Claude write
   one question plus a reference answer per chunk → `eval/questions.draft.jsonl`.
2. **Review them by hand:** `python -m eval.review_eval` shows each source passage with its question and lets
   you keep, edit or drop it → `eval/questions.jsonl`. Aim to keep 50–100. This file is the most valuable thing
   in the repo; don't skip the review.
3. **Score:** `python -m eval.run_eval --label baseline` → per-question CSV in `eval/results/` and one summary
   row appended to `eval/results/summary.csv`. Use `--limit 5` for a quick smoke test.

Metrics per run:

| Metric | Meaning |
|---|---|
| `answer_accuracy` | Share of answers the LLM judge marks correct against the reference answer |
| `doc_hit_rate` | Share of questions where the source document is in the top k |
| `chunk_hit_rate` | Same, for the exact source chunk (stricter) |
| `citation_rate` | Share of answers with at least one citation |
| `p50_ms`, `p95_ms` | End-to-end latency (retrieval + generation) |
| `cost_per_request_usd` | Answer-model cost from token usage and the price table in `rag/config.py` (judge cost reported separately) |

Run it after every change and compare rows in `summary.csv`.

> `doc_hit_rate` undercounts a little: some facts appear in more than one article (for example, Tereshkova is
> covered in both her own article and "Space Race"), and retrieving the other article still counts as a miss.
> `answer_accuracy` doesn't have that problem.

## Results

| Run | Questions | Accuracy | Doc hit | Chunk hit | p50 | p95 | Cost / request |
|---|---|---|---|---|---|---|---|
| baseline | 78 | 78.2% | 96.2% | 70.5% | 2.3 s | 4.1 s | $0.0089 |

Baseline: Claude Sonnet 5, bge-small embeddings, 450/75-token chunks, top 5. All 17 misses had the source chunk
missing from the top 5; in 15 of them the model said it didn't know rather than guessing. Retrieval is the
bottleneck, not generation.

## Storage modes

By default Qdrant runs embedded from `data/qdrant/`, so no server is needed. Only one process can open it at a
time, so stop the API before running evals. To run both at once, start the server with `docker compose up -d`,
set `QDRANT_URL=http://localhost:6333` in `.env` and re-run `python -m rag.ingest`.

## Layout

```
rag/        config, chunking, embedding store, ingest, pipeline, FastAPI app
scripts/    fetch_corpus.py (Wikipedia download)
eval/       build_eval.py, review_eval.py, run_eval.py, questions*.jsonl, results/
data/       corpus/ (committed), qdrant/ (generated)
```
