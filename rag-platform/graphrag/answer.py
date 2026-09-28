"""Short-answer generation for HotpotQA (answers are scored by exact match and F1, so they must be spans, not
sentences), plus the question-decomposition extension."""
from pydantic import BaseModel

from rag import config
from rag.pipeline import claude

from .corpus import paragraphs
from .retrieve import Retrieved, retrieve

ANSWER_PROMPT = """Answer the question using only the paragraphs below{triples_note}.

{context}

Question: {question}

Think briefly about which facts connect, then give the final answer as a short span: a name, date, number, place,
or "yes"/"no". No full sentences in the answer field. If the paragraphs don't contain it, give your best guess."""


class ShortAnswer(BaseModel):
    reasoning: str
    answer: str


def format_context(r: Retrieved) -> str:
    paras = paragraphs()
    text = "\n\n".join(f"[{t}]\n{paras[t].text}" for t in r.titles)
    if r.triples:
        text += "\n\nKnown relations:\n" + "\n".join(f"- {a} {rel} {b}" for a, rel, b in r.triples)
    return text


def answer(question: str, r: Retrieved, model: str = config.LARGE_MODEL) -> tuple[str, float]:
    resp = claude().messages.parse(
        model=model, max_tokens=2048, output_format=ShortAnswer,
        messages=[{"role": "user", "content": ANSWER_PROMPT.format(
            context=format_context(r), question=question,
            triples_note=" and the listed relations" if r.triples else "")}])
    text = resp.parsed_output.answer.strip() if resp.stop_reason == "end_turn" else ""
    return text, config.cost_usd(model, resp.usage.input_tokens, resp.usage.output_tokens)


# ---- Question decomposition (extension) ----

class Plan(BaseModel):
    sub_questions: list[str]


DECOMPOSE_PROMPT = """Split this question into the smallest sequence of simple questions that answers it, at most 3.
Later questions may refer to an earlier answer as #1, #2. If it is already simple, return it unchanged as one item.

Question: {question}"""


def decomposed_answer(question: str, graph, linker, model: str = config.LARGE_MODEL) -> tuple[str, float, Retrieved]:
    """Answer sub-questions in order, substituting earlier answers into later ones, then answer the original
    question over the union of everything retrieved along the way."""
    plan = claude().messages.parse(model=config.SMALL_MODEL, max_tokens=1024, output_format=Plan,
                                   messages=[{"role": "user", "content": DECOMPOSE_PROMPT.format(question=question)}])
    cost = config.cost_usd(config.SMALL_MODEL, plan.usage.input_tokens, plan.usage.output_tokens)
    subs = (plan.parsed_output.sub_questions or [question])[:3]
    answers, titles = [], []
    for sub in subs:
        for i, prev in enumerate(answers, start=1):
            sub = sub.replace(f"#{i}", prev)
        r = retrieve(sub, "graph", graph, linker)
        titles += r.titles
        a, c = answer(sub, r, config.SMALL_MODEL)
        answers.append(a)
        cost += c
    final = Retrieved(list(dict.fromkeys(titles))[:8], graph.triples(list(dict.fromkeys(titles))[:8]))
    text, c = answer(question, final, model)
    return text, cost + c, final
