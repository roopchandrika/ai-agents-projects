"""The HotpotQA sample and its paragraph corpus. Each paragraph is one chunk, identified by its Wikipedia title."""
import json
from dataclasses import dataclass
from functools import lru_cache

from rag import config


@dataclass(frozen=True)
class Paragraph:
    title: str
    sentences: tuple[str, ...]

    @property
    def text(self) -> str:
        return "".join(self.sentences).strip()


@lru_cache(maxsize=1)
def questions() -> list[dict]:
    path = config.HOTPOT_DIR / "sample.json"
    if not path.exists():
        raise SystemExit("No HotpotQA sample. Run: python -m scripts.fetch_hotpotqa")
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def paragraphs() -> dict[str, Paragraph]:
    """All context paragraphs across the sampled questions, pooled into one corpus (deduplicated by title)."""
    out: dict[str, Paragraph] = {}
    for q in questions():
        for title, sentences in q["context"]:
            out.setdefault(title, Paragraph(title, tuple(sentences)))
    return out


def gold_titles(question: dict) -> set[str]:
    return {title for title, _ in question["supporting_facts"]}
