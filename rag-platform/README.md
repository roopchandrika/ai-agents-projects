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

## Project 1: semantic caching and cost-aware routing

Every request (API or eval) goes through [rag/service.py](rag/service.py) and is logged to `data/requests.db`
with its cost, latency, cache result, route and model:

```
cache hit? ──yes──> verified same answer? ──yes──> stored answer   (tenant scope: user must still be able to read every source)
    │ no                                           
retrieve top 5 ──> route: reasoning words or weak retrieval? ──> Sonnet 5
                                   │ no
                                Haiku 4.5 ──> weak answer? ──> Sonnet 5 with top 10
```

**Semantic cache ([rag/cache.py](rag/cache.py)).** Normalized question → embedding → nearest entry in the same
scope, built from the same corpus version, and not past its TTL. Errors, refusals, uncited answers and "I don't
know" answers are never cached. Tenant entries are only served if the asking user can read every source chunk
*right now*, so a permission revoke also invalidates cached answers, and cache hits are written to the audit log.

**Tuning the threshold** (`python -m eval.build_cache_pairs`, then `python -m eval.tune_threshold --verify`).
There are 78 paraphrase pairs that should hit the cache and 78 near-misses that must not (another entity, attribute or
qualifier: "Pioneer 10" vs "Pioneer 11", "June 30" vs "September 30").

| Finding | Result |
|---|---|
| Median embedding similarity, paraphrases vs near-misses | 0.922 vs **0.939**: near-misses score *higher* |
| Lowest threshold with zero wrong answers, embeddings alone | 0.995, which serves 0% of paraphrases |
| Duplicate-question cross-encoders (Quora-trained) | still 9–17 false hits at usable recall |
| **Embedding ≥ 0.87 nominates, Haiku 4.5 verifies "same answer?"** | **93.6% of paraphrases served, 0 of 68 near-misses** |
| Verifier cost | $0.0004 per candidate (about 4% of a Sonnet answer); identical questions skip it |

One generated near-miss turned out to have the same answer as its original, so it was corrected by hand
(noted in `eval/cache_pairs.jsonl`).

**Routing ([rag/router.py](rag/router.py), tuned with `python -m eval.route_study`).** The study ran Haiku and
Sonnet on the 78 single-fact questions plus 28 two-article comparison and "why/how" questions
([eval/build_complex.py](eval/build_complex.py)). It labelled a question "complex" only when Sonnet got it right
and Haiku didn't, then simulated each policy from the recorded answers:

| Policy (106 questions) | Accuracy | Cost / question | Touches Sonnet |
|---|---|---|---|
| Always Sonnet (baseline) | 76.4% | $0.0096 | 100% |
| Always Haiku | 72.6% | $0.0034 | 0% |
| First rules (reasoning words, >22 words, multi-part, weak retrieval) | 76.4% | $0.0082 | 76% |
| Escalation to Sonnet with the same 5 chunks | fixed 1 of 16 | | |
| **Reasoning-words rule + escalation to Sonnet with top 10** | **84.0%** | **$0.0075** | 45% |

- **Haiku matches Sonnet on single-fact lookups.** Only 5 of 106 questions needed Sonnet, and all 5 contain
  reasoning words ("why", "compare", "how does"), so the length and multi-part rules were dropped.
- **"I don't know" usually means the answer chunk wasn't retrieved.** A bigger model alone can't fix that, so
  escalation also widens retrieval, which fixes 11 of 16.
- **Caveat:** the routing rule was chosen on the same 5 questions it's scored on, so treat the routing gain as
  optimistic. The escalation change wasn't tuned.

**Dashboard.** `streamlit run dashboard/app.py` shows cost per request, cache hit rate, how requests were served,
cumulative cost, accuracy per path and latency for any runs in the request log.

**End-to-end comparison (pending).** The same 184 requests (106 questions plus 78 paraphrased repeats) through both
paths:

```bash
python -m eval.run_eval --mode baseline --label p1-baseline --questions eval/questions.jsonl eval/questions_complex.jsonl --paraphrases
python -m eval.run_eval --mode smart --label p1-cache-routing --questions eval/questions.jsonl eval/questions_complex.jsonl --paraphrases
```

| Run | Requests | Accuracy | Cache hits | Cost / request | p50 | p95 |
|---|---|---|---|---|---|---|
| p1-baseline | – | – | – | – | – | – |
| p1-cache-routing | – | – | – | – | – | – |

## Project 3: GraphRAG for multi-hop questions

150 questions sampled from the HotpotQA dev set (distractor setting): 123 "bridge" questions ("Where was the director
of film X born?") and 27 comparisons. All their context paragraphs are pooled into **one corpus of 1,491
paragraphs**, so retrieval has to find the 2 supporting paragraphs among ~1,500, not among each question's own 10.

**Pipeline** ([graphrag/](graphrag/))

1. `python -m scripts.fetch_hotpotqa`: download (from the hotpotqa organisation's Hugging Face copy) and sample.
2. `python -m graphrag.vectors`: embed the paragraphs into Qdrant.
3. `python -m graphrag.extract`: Haiku 4.5 extracts entities and relations per paragraph into a fixed Pydantic
   schema (6 entity types, 19 relation types; unknown types fail validation). Results are appended to
   `data/hotpotqa/extractions.jsonl` as they arrive, so the run can resume and is paid for once. Estimated cost:
   about $3–4.
4. `docker compose up -d neo4j`, then `python -m graphrag.graph`: entity resolution, then load
   `(:Entity)-[:REL {type, chunk}]->(:Entity)` and `(:Chunk)-[:MENTIONS]->(:Entity)` into Neo4j.
5. `python -m eval.hotpot_eval --mode {vector, vector_rerank, graph, graph_decompose}`: scores exact match, F1,
   and supporting-paragraph recall, split into bridge and comparison. `--retrieval-only` skips the model and
   costs nothing.

**Entity resolution** ([graphrag/resolve.py](graphrag/resolve.py)) merges names with union-find in three steps:
- **Normalized names:** case, accents, punctuation, a leading "the" and a "(film)"-style suffix are ignored.
  Disambiguated Wikipedia titles such as "Mercury (planet)" and "Mercury (element)" are never merged.
- **Person initials:** "J. R. Smith" joins "John Robert Smith" when only one full name fits.
- **Embedding similarity (≥ 0.93):** only between names that share a word and have **identical numbers**. That
  guard is the lesson from Project 1: "Apollo 11" and "Apollo 13" look alike to an embedding model.

Paragraph titles are Wikipedia article names, so a group containing one takes it as the canonical name.

**Retrieval** ([graphrag/retrieve.py](graphrag/retrieve.py)). Every mode gives the model 5 paragraphs, so modes
differ only in *which* paragraphs they pick:
- **Graph mode** links entities named in the question (whole-word match against names and aliases) and takes the
  entities in the top 2 vector hits (the bridge entity is usually in the first-hop paragraph).
- It expands them up to 2 hops in Cypher. Hub entities with more than 60 links aren't traversed, so "United States"
  doesn't connect everything to everything.
- Paragraphs mentioning the entities it reaches become candidates, and the relation triples between the chosen
  paragraphs go into the prompt.
- **Choosing the final 5:** `GRAPH_STRATEGY=rerank` (the default) reranks vector and graph candidates together.
  `slots` keeps the top 3 vector hits and fills 2 slots from the graph.
- **`graph_decompose`** splits the question into up to 3 sub-questions and answers them in order, substituting
  earlier answers. It sees up to 8 paragraphs, so compare it with that in mind.

**Measured so far: retrieval only, 150 questions, no API cost**

| Mode | Supporting-paragraph recall | Both gold paragraphs found | Bridge | Comparison | p50 |
|---|---|---|---|---|---|
| vector (top 5) | 90.0% | 80.0% | 76.4% | 96.3% | 0.03 s |
| + ms-marco-MiniLM-L-6 reranker | 85.7% | 72.7% | 68.3% | 92.6% | 1.1 s |
| + ms-marco-MiniLM-L-12 reranker | 85.3% | 72.0% | 66.7% | 96.3% | 2.2 s |
| **+ bge-reranker-base** | **92.3%** | **84.7%** | **81.3%** | **100%** | 7.9 s |
| graph | pending extraction | | | | |

- **Reranker choice decides whether reranking helps at all.** The MS MARCO cross-encoders are 7–8 points *worse*
  than no reranker. bge-reranker-base is 4.7 points better, so it's the default, and `vector_rerank` with it is
  the bar the graph has to beat.
- **Bridge questions are the gap:** 81% vs 96% for comparisons.

**Still to run (needs API credit and Neo4j):** extraction; the single-hop set
(`python -m eval.build_hotpot_single`, then review), to check easy questions don't get worse; and the answer-level
runs (EM/F1) for every mode. Expect roughly $6–8 in total.

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
- Chat history and summaries would also need ACLs; there are none yet. The Project 1 cache is scoped per
  tenant and re-checks every source against the asking user's current permissions.

## Storage modes

By default Qdrant runs embedded from `data/qdrant/`, so no server is needed. Only one process can open it at a
time, so stop the API before running evals. To run both at once, start the server with `docker compose up -d`,
set `QDRANT_URL=http://localhost:6333` in `.env` and re-run `python -m rag.ingest`.

## Layout

```
rag/        config, chunking, embedding store, ingest, pipeline, FastAPI app,
            acl, auth, audit, secure retrieval, tenant ingest (Project 2),
            service, cache, router, request log (Project 1)
graphrag/   HotpotQA corpus, vectors, extraction, entity resolution, Neo4j graph, retrieval, answering (Project 3)
scripts/    fetch_corpus.py, issue_token.py, measure_acl.py, fetch_hotpotqa.py
eval/       build_eval.py, review_eval.py, run_eval.py, leakage_cases.py, questions*.jsonl, results/
tests/      leakage suite, auth, audit, revoke, ingest validation, cache and routing, graphrag (pytest)
dashboard/  Streamlit cost & quality dashboard over data/requests.db
data/       corpus/, tenants/, hotpotqa/sample.json (committed); qdrant/, audit.db, requests.db,
            hotpotqa/raw/ (generated)
```
