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

## Project 2: permission-aware multi-tenant RAG

Three fictional companies (Northwind Robotics, Helix Biotherapeutics, Summit Freight) share one index. Each has
4 users and 8 documents ranging from the employee handbook to board minutes, salary bands and individual
performance reviews. Every company has the same kinds of sensitive documents, so a query like "salary bands" is a
realistic cross-tenant attack. The Wikipedia corpus is split across the companies as filler, bringing the index to
2,070 chunks.

**How access is enforced**

1. **Every chunk carries its ACL**: `tenant_id`, `allowed_roles`, `allowed_users`, `classification`. Ingestion
   validates each one and refuses to write a chunk with no tenant ([rag/acl.py](rag/acl.py),
   [rag/tenant_ingest.py](rag/tenant_ingest.py)).
2. **Identity comes only from a signed JWT** ([rag/auth.py](rag/auth.py)). The request body has no tenant or role
   fields; extra fields are ignored. Forged, unsigned and expired tokens get a 401.
3. **Filtering happens inside the vector search** (pre-filtering), so forbidden chunks never compete for the top k.
   Post-filtering (retrieve, then drop) would leave users with fewer or empty results and reveal through missing
   results that something was there.
4. **A second check in application code** re-verifies every chunk before it reaches the prompt. Failures are
   dropped and logged as security alerts ([rag/secure.py](rag/secure.py)).
5. **Append-only audit log** of every retrieval, ACL change and alert. SQLite triggers reject `UPDATE` and
   `DELETE` ([rag/audit.py](rag/audit.py)).
6. **ACL changes apply to every chunk of a document at once**, through an admin endpoint scoped to the admin's own
   tenant.

**Try it**

```bash
python -m rag.tenant_ingest                          # seed docs + filler -> 'tenant_chunks'
uvicorn rag.api:app --reload
python -m scripts.issue_token --list                 # the 12 seed users and their roles
python -m scripts.issue_token nw-alice               # prints a JWT (needs JWT_SECRET in .env)
```

| Endpoint | Who | What |
|---|---|---|
| `POST /tenant/ask` | any user | Answer from documents the token's user may read |
| `PUT /admin/documents/{doc_id}/acl` | tenant admin | Change a document's ACL |
| `GET /admin/documents/{doc_id}/access-log?days=30` | tenant admin | Who retrieved this document |
| `GET /admin/alerts` | tenant admin | Chunks the second check had to block |

**Tests** (`pytest`, run in CI on every push; no API key needed)

- **Leakage suite:** 1,024 adversarial queries across all 12 users. Each user fires 4 queries at every document
  they may not read (its title, its distinctive fact, and prompt-injection phrasings like "ignore your filters"),
  plus 10 generic attacks. The test asserts that zero forbidden chunks appear in the top 10.
  [eval/leakage_cases.py](eval/leakage_cases.py) checks access against the manifest, not the stored payloads, so a
  payload bug can't hide a leak.
- **Control test:** the same targeted queries run without the filter reach their forbidden document 100% of the
  time, which proves the suite would catch a leak.
- **Reachability test:** every user can still retrieve every document they're allowed to read. A filter that
  returns nothing would fail it.
- **Other tests:**
  - forged, unsigned and expired tokens are rejected, and tenant or roles in the body are ignored
  - admins can't reach other tenants' documents
  - the audit log is append-only
  - a broken filter is caught by the second check
  - a revoke takes effect on the very next query
- **Opt-in end-to-end test** with the real model (`RUN_LLM_TESTS=1 pytest tests/test_llm_leakage.py`, about
  $0.25): 24 injection attempts, and no forbidden fact appears in any answer.

**Measurements** (`python -m scripts.measure_acl` → [eval/results/acl_metrics.json](eval/results/acl_metrics.json))

| Metric | Result |
|---|---|
| Cross-tenant or cross-role leaks | **0** of 1,024 adversarial queries (top 10 checked) |
| Stale reads after a revoke | **0** of 20 (the very next query never returned the document) |
| Revoke call to enforced | 65 ms median, 154 ms max (includes the audit write) |
| Filter overhead per search | 22 ms median, **embedded mode**; see the note below |

> Embedded Qdrant evaluates filters in Python and ignores payload indexes, so the 22 ms overstates the cost. For a
> representative number, start Docker, run `docker compose up -d`, set `QDRANT_URL=http://localhost:6333`,
> re-run `python -m rag.ingest` and `python -m rag.tenant_ingest`, then `python -m scripts.measure_acl`.

**Limitations**

- ACL changes made through the API live in the vector store and the audit log. Re-running `tenant_ingest`
  resets them to the manifest, which is the source of truth at ingest time.
- There is no login flow: tokens are minted by a script for the seed users.
- Chat history, summaries and caches would also need ACLs. There are none yet; Project 1's cache must key on
  tenant and user.

## Storage modes

By default Qdrant runs embedded from `data/qdrant/`, so no server is needed. Only one process can open it at a
time, so stop the API before running evals. To run both at once, start the server with `docker compose up -d`,
set `QDRANT_URL=http://localhost:6333` in `.env` and re-run `python -m rag.ingest`.

## Layout

```
rag/        config, chunking, embedding store, ingest, pipeline, FastAPI app,
            acl, auth, audit, secure retrieval, tenant ingest (Project 2)
scripts/    fetch_corpus.py, issue_token.py, measure_acl.py
eval/       build_eval.py, review_eval.py, run_eval.py, leakage_cases.py, questions*.jsonl, results/
tests/      leakage suite, auth, audit, revoke, ingest validation (pytest)
data/       corpus/ and tenants/ (committed), qdrant/ and audit.db (generated)
```
