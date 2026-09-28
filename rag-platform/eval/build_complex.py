"""Draft harder eval questions that need two related articles: comparisons and "why/how" explanations.

The main eval set is single-fact lookups, which a small model handles; these give the router something to
route. Usage: python -m eval.build_complex  -> eval/questions_complex.jsonl (review before use)
"""
import json
import random
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel

from rag import config
from rag.corpus import chunk_all, load_documents
from rag.pipeline import claude
from rag.store import embedder

OUT_PATH = config.EVAL_DIR / "questions_complex.jsonl"
DOC_PAIRS = [
    ("spirit-rover", "opportunity-rover"), ("voyager-1", "voyager-2"), ("pioneer-10", "pioneer-11"),
    ("apollo-11", "apollo-13"), ("hubble-space-telescope", "james-webb-space-telescope"),
    ("falcon-9", "falcon-heavy"), ("mir", "international-space-station"), ("chang-e-4", "chang-e-5"),
    ("chandrayaan-2", "chandrayaan-3"), ("curiosity-rover", "perseverance-rover"),
    ("space-shuttle-challenger-disaster", "space-shuttle-columbia-disaster"), ("hayabusa2", "osiris-rex"),
    ("kepler-space-telescope", "transiting-exoplanet-survey-satellite"), ("yuri-gagarin", "valentina-tereshkova"),
    ("saturn-v", "space-launch-system"), ("ariane-5", "ariane-6"), ("rosetta-spacecraft", "philae-spacecraft"),
    ("cassini-huygens", "galileo-spacecraft"), ("sputnik-1", "space-race"), ("skylab", "salyut-1"),
    ("mars-express", "mars-reconnaissance-orbiter"), ("tianwen-1", "zhurong-rover"),
    ("parker-solar-probe", "solar-orbiter"), ("messenger", "bepicolombo"), ("buran-spacecraft", "energia-rocket"),
    ("spacex-dragon-2", "commercial-crew-program"), ("wernher-von-braun", "sergei-korolev"),
    ("neil-armstrong", "buzz-aldrin"), ("project-mercury", "project-gemini"), ("gravity-assist", "new-horizons"),
]
KINDS = ["a comparison that needs a fact from each passage", "a why/how explanation that combines both passages"]

PROMPT = """Here are passages from two Wikipedia articles.

<passage title="{t1}">
{p1}
</passage>

<passage title="{t2}">
{p2}
</passage>

Write one question for a space-exploration Q&A assistant: {kind}. It must need information from BOTH passages to \
answer fully, make sense on its own (name the missions or people, never say "the passage"), and have a clear, \
checkable answer. Then give the reference answer in two to four sentences, using only the passages."""


class Item(BaseModel):
    question: str
    reference_answer: str


def main() -> None:
    rng = random.Random(7)
    chunks = [c for c in chunk_all(load_documents(), embedder().tokenizer) if len(c.text) >= 800]
    by_doc = {}
    for c in chunks:
        by_doc.setdefault(c.doc_id, []).append(c)

    jobs = []
    for i, (a, b) in enumerate(DOC_PAIRS):
        ca, cb = rng.choice(by_doc[a][:4]), rng.choice(by_doc[b][:4])  # early chunks: overview facts
        jobs.append((ca, cb, KINDS[i % 2]))

    def draft(job):
        ca, cb, kind = job
        return claude().messages.parse(
            model=config.EVAL_GEN_MODEL, max_tokens=16000, output_format=Item,
            messages=[{"role": "user", "content": PROMPT.format(t1=ca.title, p1=ca.text, t2=cb.title, p2=cb.text,
                                                                kind=kind)}]).parsed_output

    with ThreadPoolExecutor(max_workers=4) as pool:
        items = list(pool.map(draft, jobs))
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for i, ((ca, cb, kind), it) in enumerate(zip(jobs, items), start=1):
            f.write(json.dumps({"id": f"c{i:03d}", "question": it.question, "reference_answer": it.reference_answer,
                                "doc_id": ca.doc_id, "chunk_id": ca.chunk_id, "doc_ids": [ca.doc_id, cb.doc_id],
                                "chunk_ids": [ca.chunk_id, cb.chunk_id], "kind": kind.split(" ")[1]},
                               ensure_ascii=False) + "\n")
    print(f"Wrote {len(items)} questions to {OUT_PATH}")


if __name__ == "__main__":
    main()
