"""Entity and relation extraction with a fixed schema.

Each paragraph goes to the extraction model once, with structured output validated against the Pydantic models
below. Entity and relation types are closed sets (Literal), so the model can't invent new ones; a small fixed
schema gives a much cleaner graph than free-form extraction.

Results are appended to data/hotpotqa/extractions.jsonl as they arrive, so a run can be interrupted and resumed,
and the (committed) file means nobody has to pay for extraction twice.

Usage: python -m graphrag.extract [--limit N] [--workers 8]
"""
import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

import anthropic
from pydantic import BaseModel, Field

from rag import config
from rag.pipeline import claude

from .corpus import Paragraph, paragraphs

EXTRACTIONS = config.HOTPOT_DIR / "extractions.jsonl"

EntityType = Literal["PERSON", "ORGANIZATION", "LOCATION", "WORK", "EVENT", "OTHER"]
RelationType = Literal[
    "DIRECTED", "PRODUCED", "WROTE", "PERFORMED_IN", "CREATED", "FOUNDED", "MEMBER_OF", "EMPLOYED_BY",
    "BORN_IN", "DIED_IN", "LOCATED_IN", "PART_OF", "CITIZEN_OF", "SPOUSE_OF", "PARENT_OF", "PUBLISHED_BY",
    "AWARDED", "NAMED_AFTER", "RELATED_TO",
]


class Entity(BaseModel):
    name: str = Field(description="Full name as written in the paragraph, e.g. 'John Smith', 'Appenzell'")
    type: EntityType


class Relation(BaseModel):
    source: str = Field(description="Name of an entity from the entities list")
    relation: RelationType
    target: str = Field(description="Name of an entity from the entities list")


class Extraction(BaseModel):
    entities: list[Entity]
    relations: list[Relation]


PROMPT = """Extract a knowledge graph from this Wikipedia paragraph.

<paragraph title="{title}">
{text}
</paragraph>

Entities: the paragraph's subject ("{title}") plus every named person, organization, place, work (film, book,
album, song, show, game), event or other named thing it mentions. Use the fullest name the paragraph gives.

Relations: facts stated in the paragraph that connect two of those entities, oriented source -> target:
DIRECTED/PRODUCED/WROTE (person -> work), PERFORMED_IN (person -> work), CREATED/FOUNDED (person or org -> thing),
MEMBER_OF, EMPLOYED_BY, BORN_IN, DIED_IN, LOCATED_IN, PART_OF, CITIZEN_OF, SPOUSE_OF, PARENT_OF (parent -> child),
PUBLISHED_BY (work -> org), AWARDED (entity -> award), NAMED_AFTER. Use RELATED_TO only when none of these fit.
Only include relations the paragraph actually states."""


def extract(p: Paragraph) -> dict:
    r = claude().messages.parse(
        model=config.EXTRACT_MODEL, max_tokens=4096, output_format=Extraction,
        messages=[{"role": "user", "content": PROMPT.format(title=p.title, text=p.text)}])
    ex = r.parsed_output if r.stop_reason == "end_turn" else Extraction(entities=[], relations=[])
    return {"title": p.title, "entities": [e.model_dump() for e in ex.entities],
            "relations": [rel.model_dump() for rel in ex.relations], "stop_reason": r.stop_reason,
            "input_tokens": r.usage.input_tokens, "output_tokens": r.usage.output_tokens,
            "cost_usd": config.cost_usd(config.EXTRACT_MODEL, r.usage.input_tokens, r.usage.output_tokens)}


def load() -> dict[str, dict]:
    if not EXTRACTIONS.exists():
        return {}
    rows = [json.loads(line) for line in EXTRACTIONS.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {r["title"]: r for r in rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="extract at most N more paragraphs (quick checks)")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    done = load()
    todo = [p for t, p in paragraphs().items() if t not in done]
    todo = todo[:args.limit] if args.limit else todo
    print(f"{len(done)} paragraphs already extracted, {len(todo)} to go with {config.EXTRACT_MODEL}")

    lock, spent, failed = threading.Lock(), [0.0], []

    def work(p: Paragraph) -> None:
        try:
            row = extract(p)
        except anthropic.APIStatusError as e:  # keep going; failures are retried on the next run
            failed.append((p.title, e.status_code))
            return
        with lock, EXTRACTIONS.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            spent[0] += row["cost_usd"]

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(work, todo))
    total = sum(r["cost_usd"] for r in load().values())
    print(f"Done. This run: ${spent[0]:.3f}; total extraction cost so far: ${total:.3f}")
    if failed:
        print(f"{len(failed)} failed (rerun to retry): {failed[:5]}")


if __name__ == "__main__":
    main()
