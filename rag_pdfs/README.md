# rag-pdf-chatbot

A retrieval-augmented generation (RAG) chatbot that answers questions
grounded in a set of PDF documents, citing the source file and page number
for each part of its answer.

## Pipeline

```
PDF(s) --[pypdf: extract text per page]--> page text
       --[chunk_text: ~800 char chunks, 150 char overlap]--> chunks
       --[embed]--> vectors --[chromadb: persistent local store]--> index

query --[embed with the SAME fitted embedder]--> vector
      --[chromadb similarity search]--> top-k chunks (+ source/page metadata)
      --[build_context: tag each chunk with [source, p.N]]--> context
      --[Claude, instructed to cite sources]--> final answer
```

## A real, verified run

I generated a small 3-page sample PDF (`make_sample_pdf.py`) covering a
fictional company's travel, reimbursement, and remote-work policies, ran it
through the real ingestion pipeline, and queried it:

```
$ python ingest.py sample_docs/employee_handbook.pdf
Ingested 3 chunks from 1 PDF(s) into ./chroma_db

$ python3 -c "from retrieve import load_collection, retrieve; ..."
QUERY: How many days can I work from home?
  -> [employee_handbook.pdf, p.3] (distance=1.168) Remote Work Guidelines Employees may work remotely up to 3 days per week...
  -> [employee_handbook.pdf, p.1] (distance=1.796) Company Travel Policy All employees traveling for business must...
```

The correct page was ranked first. This was a genuine run against a real
PDF and a real Chroma index — not a mocked result.

## Important, honest limitation: the default embedder is TF-IDF, not neural

I built this with **scikit-learn's TF-IDF** as the default embedder
(`embeddings.py`), not a neural embedding model. This was a deliberate
trade-off for this environment (no large model downloads, fast, fully
local, easy to test), but it means retrieval is based on **word overlap**,
not semantic meaning — it won't match "car" to "automobile" or "PTO" to
"paid time off" the way a neural embedding model would.

`embeddings.py` defines a small `Embedder` protocol (`fit`, `embed`)
specifically so this can be swapped without touching `ingest.py` or
`retrieve.py`. The commented-out `NeuralEmbedder` class in that file shows
the swap to `sentence-transformers`; Voyage AI (Anthropic's recommended
embeddings partner) is another drop-in option for production use. I'd
treat that swap as the single highest-value next step for this project if
retrieval quality on real, larger documents turned out to matter more than
what TF-IDF gives you.

## Why the embedder is pickled at ingest time

TF-IDF vectors are only meaningful relative to the vocabulary they were
fit on. `ingest.py` fits the vectorizer once over the whole corpus and
**pickles the fitted embedder** into the persistence directory alongside
the Chroma collection, so `chat.py` embeds new queries with the exact same
fitted vectorizer rather than a fresh, incompatible one.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then add your ANTHROPIC_API_KEY
```

## Usage

```bash
# 1. Build the index from one or more PDFs
python ingest.py sample_docs/employee_handbook.pdf
# (or generate the sample PDF yourself first: python make_sample_pdf.py)

# 2. Ask questions
python chat.py "How much can I spend on a hotel per night?"
python chat.py   # interactive mode
```

## Tests

```bash
pip install pytest
python -m pytest test_rag.py -v
```

Covers chunk boundaries/overlap, citation formatting, and end-to-end
retrieval against a real (temporary) Chroma collection built from a real
generated PDF. None of this requires an API key — only `chat.py`'s final
generation step calls Claude.

## What I'd add next

- Swap in a neural embedder (see above) and add a small retrieval-quality
  eval comparing TF-IDF vs. neural on a fixed set of test questions —
  this is exactly the kind of before/after comparison the "hybrid search"
  follow-on project (keyword + semantic) is meant to produce.
- Re-rank retrieved chunks with a cross-encoder before building context,
  since first-pass retrieval (TF-IDF or neural) is a recall step, not
  necessarily a precision one.
- Guard against citing a source the model wasn't actually given — right
  now the system prompt asks Claude to only use the provided context, but
  nothing programmatically verifies a cited `[file, page]` pair was in the
  retrieved set.

## Project structure

```
rag-pdf-chatbot/
├── ingest.py           # PDF text extraction, chunking, embedding, Chroma storage
├── retrieve.py          # query embedding, similarity search, citation formatting
├── chat.py              # RAG loop: retrieve -> build context -> ask Claude
├── embeddings.py         # swappable Embedder (TF-IDF by default)
├── make_sample_pdf.py    # generates a demo PDF so the repo is runnable out of the box
├── sample_docs/          # generated sample PDF lives here
├── test_rag.py           # chunking, citation, and real end-to-end retrieval tests
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```
