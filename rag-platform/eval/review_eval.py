"""Hand-review drafted questions: keep, edit or drop each one.

Usage: python -m eval.review_eval
Reads eval/questions.draft.jsonl and appends approved items to eval/questions.jsonl.
You can stop at any time (q) and resume later; already-reviewed ids are skipped.
"""
import json
import textwrap

from rag import config

DRAFT_PATH = config.EVAL_DIR / "questions.draft.jsonl"
FINAL_PATH = config.EVAL_DIR / "questions.jsonl"
DROPPED_PATH = config.EVAL_DIR / "questions.dropped.jsonl"

CHECKLIST = """Drop it if the question is ambiguous, needs the passage to make sense, has more than one
defensible answer, or the reference answer is wrong or incomplete."""


def reviewed_ids() -> set[str]:
    ids = set()
    for path in (FINAL_PATH, DROPPED_PATH):
        if path.exists():
            ids |= {json.loads(line)["id"] for line in path.read_text(encoding="utf-8").splitlines() if line}
    return ids


def append(path, item: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> None:
    items = [json.loads(line) for line in DRAFT_PATH.read_text(encoding="utf-8").splitlines() if line]
    done = reviewed_ids()
    todo = [it for it in items if it["id"] not in done]
    print(f"{len(items) - len(todo)} already reviewed, {len(todo)} to go.\n{CHECKLIST}\n")

    for n, item in enumerate(todo, start=1):
        print("=" * 80)
        print(f"[{n}/{len(todo)}] {item['id']}  source: {item['title']} ({item['chunk_id']})\n")
        print(textwrap.indent(textwrap.fill(item["source_text"], 100), "  | "))
        print(f"\nQ: {item['question']}\nA: {item['reference_answer']}\n")
        while True:
            choice = input("[k]eep  [e]dit  [d]rop  [q]uit > ").strip().lower()
            if choice == "k":
                append(FINAL_PATH, item)
            elif choice == "d":
                append(DROPPED_PATH, item)
            elif choice == "e":
                item["question"] = input("Question (enter to keep): ").strip() or item["question"]
                item["reference_answer"] = input("Answer (enter to keep): ").strip() or item["reference_answer"]
                append(FINAL_PATH, item)
            elif choice == "q":
                return
            else:
                continue
            break

    kept = len(FINAL_PATH.read_text(encoding="utf-8").splitlines()) if FINAL_PATH.exists() else 0
    print(f"\nDone. {kept} questions in {FINAL_PATH}")


if __name__ == "__main__":
    main()
