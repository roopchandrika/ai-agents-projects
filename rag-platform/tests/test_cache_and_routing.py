"""Semantic cache correctness (scoping, permissions, invalidation) and routing/escalation, with a fake model."""
import pytest

from rag import cache, config, pipeline, router, service

REVIEW = "northwind/perf-review-alice-chen"


@pytest.fixture
def fresh_cache():
    cache.clear()
    yield
    cache.clear()


@pytest.fixture
def model(monkeypatch):
    """Fake generate(): answers by model name, recording each call. Tests set `replies` per model."""
    calls, replies = [], {}

    def generate(question, hits, model=config.ANSWER_MODEL):
        calls.append(model)
        return replies.get(model, "Answer from the sources [1]."), "end_turn", 100, 20

    monkeypatch.setattr(pipeline, "generate", generate)
    return calls, replies


def ask(question, user, **kw):
    return service.answer(question, user, use_cache=True, use_routing=kw.pop("routing", False), **kw)


# ---- cache ----

def test_repeat_question_is_served_from_cache(fresh_cache, model, users):
    calls, _ = model
    first = ask("What is Flexday Fridays?", users["nw-alice"])
    second = ask("what is flexday fridays", users["nw-alice"])
    assert not first.cache_hit and second.cache_hit
    assert second.result.cost_usd == 0 and len(calls) == 1


def test_cached_answer_needs_access_to_every_source(fresh_cache, model, users):
    """Alice (engineering) gets sources Carol (exec) can't read, e.g. the collision postmortem, so Carol
    must not reuse Alice's answer even though it's the same question in the same tenant."""
    first = ask("PalletPilot dock collision firmware", users["nw-alice"])
    assert "northwind/postmortem-pp-collision" in {h.doc_id for h in first.result.sources}
    assert not ask("PalletPilot dock collision firmware", users["nw-carol"]).cache_hit


def test_cache_never_crosses_tenants(fresh_cache, model, users):
    ask("What are the salary bands for 2026?", users["nw-bob"])
    other = ask("What are the salary bands for 2026?", users["hx-frank"])
    assert not other.cache_hit
    assert all(h.doc_id.startswith("helix/") for h in other.result.sources)


def test_cached_answer_is_not_served_to_a_user_who_cannot_read_its_sources(fresh_cache, model, users):
    ask("Northwind salary bands 2026", users["nw-bob"])        # HR: answer built from the salary doc
    reply = ask("Northwind salary bands 2026", users["nw-alice"])  # engineer: may not read it
    assert not reply.cache_hit
    assert "northwind/salary-bands-2026" not in {h.doc_id for h in reply.result.sources}


def test_revoke_invalidates_cached_answers(fresh_cache, model, users, client, auth):
    alice = users["nw-alice"]
    assert REVIEW in {h.doc_id for h in ask("Performance Review: Alice Chen", alice).result.sources}
    assert ask("Performance Review: Alice Chen", alice).cache_hit

    revoke = {"allowed_roles": ["hr"], "allowed_users": [], "classification": "restricted"}
    restore = {"allowed_roles": ["hr"], "allowed_users": ["nw-alice"], "classification": "restricted"}
    try:
        assert client.put(f"/admin/documents/{REVIEW}/acl", headers=auth("nw-dave"), json=revoke).status_code == 200
        after = ask("Performance Review: Alice Chen", alice)
        assert not after.cache_hit
        assert REVIEW not in {h.doc_id for h in after.result.sources}
    finally:
        client.put(f"/admin/documents/{REVIEW}/acl", headers=auth("nw-dave"), json=restore)


def test_expired_entries_are_not_served(fresh_cache, model, users, monkeypatch):
    monkeypatch.setattr(config, "CACHE_TTL_HOURS", -1)
    ask("What is Flexday Fridays?", users["nw-alice"])
    assert not ask("What is Flexday Fridays?", users["nw-alice"]).cache_hit


def test_new_corpus_version_invalidates_entries(fresh_cache, model, users, monkeypatch):
    ask("What is Flexday Fridays?", users["nw-alice"])
    with monkeypatch.context() as m:  # undone before fresh_cache's teardown needs the real function
        m.setattr(cache, "corpus_version", lambda scope: "re-ingested")
        assert not ask("What is Flexday Fridays?", users["nw-alice"]).cache_hit


@pytest.mark.parametrize("reply", ["", "I don't know based on the available documents.",
                                   "An answer with no citation."])
def test_weak_answers_are_not_cached(fresh_cache, model, users, reply):
    _, replies = model
    replies[config.LARGE_MODEL] = reply
    ask("What is Flexday Fridays?", users["nw-alice"])
    assert not ask("What is Flexday Fridays?", users["nw-alice"]).cache_hit


def test_cache_hits_are_audited(fresh_cache, model, users, client, auth):
    ask("Northwind salary bands 2026", users["nw-bob"])
    ask("Northwind salary bands 2026", users["nw-carol"])  # exec, served from cache
    r = client.get("/admin/documents/northwind/salary-bands-2026/access-log", headers=auth("nw-dave"))
    assert "nw-carol" in [u["user_id"] for u in r.json()["users"]]


# ---- routing ----

class FakeHit:
    def __init__(self, score, doc_id="d"):
        self.score, self.doc_id = score, doc_id


@pytest.mark.parametrize("question,route", [
    ("What year was Helix founded?", "simple"),
    ("Why did the LIMS outage happen and how was it fixed?", "complex"),
    ("Compare the Northwind and Summit remote work policies", "complex"),
    ("How many PTO days do Northwind employees get, and what is the learning budget?", "simple"),  # lookups
])
def test_router_rules(question, route):
    hits = [FakeHit(0.8), FakeHit(0.7)]  # strong retrieval, so only the question's wording decides
    assert router.classify(question, hits).route == route


def test_weak_retrieval_routes_to_the_large_model():
    assert router.classify("What year was Helix founded?", [FakeHit(0.3)]).route == "complex"


@pytest.fixture
def lenient_router(monkeypatch):
    monkeypatch.setattr(router, "MIN_TOP_SCORE", 0.0)
    monkeypatch.setattr(router, "ESCALATE_BELOW_SCORE", 0.0)


def test_weak_small_model_answer_is_escalated(fresh_cache, model, users, lenient_router):
    calls, replies = model
    replies[config.SMALL_MODEL] = "I don't know based on the available documents."
    served = ask("What is Flexday Fridays?", users["nw-alice"], routing=True)
    assert served.route == "simple" and served.escalated
    assert calls == [config.SMALL_MODEL, config.LARGE_MODEL]
    assert served.result.model == config.LARGE_MODEL
    assert len(served.result.sources) > config.TOP_K  # escalation widened retrieval


def test_good_small_model_answer_is_kept(fresh_cache, model, users, lenient_router):
    calls, _ = model
    served = ask("What is Flexday Fridays?", users["nw-alice"], routing=True)
    assert served.route == "simple" and not served.escalated and calls == [config.SMALL_MODEL]


@pytest.mark.parametrize("text,stop,cited,why", [
    ("", "end_turn", False, "empty answer"),
    ("The sources do not mention this.", "end_turn", True, "said it doesn't know"),
    ("A fact.", "end_turn", False, "no citation"),
    ("A fact [1].", "max_tokens", True, "stop_reason=max_tokens"),
])
def test_escalation_reasons(text, stop, cited, why):
    assert router.needs_escalation(text, stop, cited, []) == why
