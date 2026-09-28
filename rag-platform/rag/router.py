"""Cost-aware routing: send simple questions to the small model and complex ones to the large model,
and escalate when the small model's answer looks weak."""
import re
from dataclasses import dataclass

REASONING_WORDS = re.compile(
    r"\b(compare|comparison|versus|vs\.?|differ|difference|different|both|why|explain|how does|how do|"
    r"what caused|impact|implications?|significance|relationship|trade-?offs?|advantages?|disadvantages?)\b", re.I)
MULTI_PART = re.compile(r",\s*and\s+(how|what|when|where|who|which|why)\b|\?\s*\w.*\?", re.I)
UNANSWERED = re.compile(r"\b(don't know|do not know|not (?:contain|mention|provide|specify|include)|"
                        r"no information|unable to (?:find|answer))\b", re.I)

# Tuned with eval/route_study.py. On this corpus the small model matches the large one on single-fact questions,
# and every question the large model got right but the small one missed contained reasoning words. Length and
# multi-part rules sent most single-fact questions to the large model for no accuracy gain, so they're logged as
# features but no longer route.
MIN_TOP_SCORE = 0.55
ESCALATE_BELOW_SCORE = 0.55
# Escalation re-retrieves more chunks: most "I don't know" answers mean the right chunk wasn't in the top 5,
# which a bigger model alone can't fix (1/16 fixed at k=5 vs 11/16 at k=10).
ESCALATION_TOP_K = 10


@dataclass
class RouteDecision:
    route: str                  # "simple" or "complex"
    reasons: list[str]


def features(question: str, hits: list) -> dict:
    scores = [h.score for h in hits] or [0.0]
    return {
        "words": len(question.split()),
        "reasoning": bool(REASONING_WORDS.search(question)),
        "multi_part": bool(MULTI_PART.search(question)),
        "top_score": scores[0],
        "score_spread": scores[0] - scores[-1],
        "distinct_docs": len({h.doc_id for h in hits}),
    }


def classify(question: str, hits: list) -> RouteDecision:
    f = features(question, hits)
    reasons = []
    if f["reasoning"]:
        reasons.append("reasoning words")
    if f["top_score"] < MIN_TOP_SCORE:
        reasons.append(f"weak retrieval ({f['top_score']:.2f})")
    return RouteDecision("complex" if reasons else "simple", reasons)


def is_unanswered(text: str) -> bool:
    return bool(UNANSWERED.search(text))


def needs_escalation(answer_text: str, stop_reason: str, cited: bool, hits: list) -> str | None:
    """Why a small-model answer should be rerun on the large model, or None if it looks fine."""
    if stop_reason != "end_turn":
        return f"stop_reason={stop_reason}"
    if not answer_text.strip():
        return "empty answer"
    if is_unanswered(answer_text):
        return "said it doesn't know"
    if not cited:
        return "no citation"
    if hits and hits[0].score < ESCALATE_BELOW_SCORE:
        return "low retrieval score"
    return None
