"""GraphRAG building blocks: entity resolution, graph expansion, question linking, metrics and the extraction
schema. Uses the in-memory graph, so no Neo4j or model calls are needed."""
import numpy as np
import pydantic
import pytest

from eval.hotpot_eval import exact_match, f1
from graphrag import retrieve as retrieve_mod
from graphrag.corpus import Paragraph
from graphrag.extract import Extraction
from graphrag.graph import MemoryGraph
from graphrag.resolve import initials_compatible, normalize, resolve


def ent(name, type_):
    return {"name": name, "type": type_}


def rel(src, r, dst):
    return {"source": src, "relation": r, "target": dst}


EXTRACTIONS = {
    "Night Harbor": {  # a film paragraph: names the director but not where he was born
        "entities": [ent("Night Harbor", "WORK"), ent("J. R. Calloway", "PERSON")],
        "relations": [rel("J. R. Calloway", "DIRECTED", "Night Harbor")]},
    "James Robert Calloway": {  # the bridge paragraph: the director's own article
        "entities": [ent("James Robert Calloway", "PERSON"), ent("Duluth, Minnesota", "LOCATION")],
        "relations": [rel("James Robert Calloway", "BORN_IN", "Duluth, Minnesota")]},
    "Apollo 11": {"entities": [ent("Apollo 11", "EVENT"), ent("the Apollo 13", "EVENT")], "relations": []},
    "Mercury (planet)": {"entities": [ent("Mercury", "LOCATION")], "relations": []},
    "Mercury (element)": {"entities": [ent("Mercury", "OTHER")], "relations": []},
    "Unrelated": {"entities": [ent("Some Band", "ORGANIZATION")], "relations": []},
}


@pytest.fixture(scope="module")
def resolved():
    return resolve(EXTRACTIONS)


def eid(resolved, name):
    return next(e["id"] for e in resolved.entities.values() if e["name"] == name or name in e["aliases"])


# ---- entity resolution ----

@pytest.mark.parametrize("raw,norm", [("The Beatles", "beatles"), ("Mercury (planet)", "mercury"),
                                      ("Pelé", "pele"), ("J.R.R. Tolkien", "j r r tolkien")])
def test_normalize(raw, norm):
    assert normalize(raw) == norm


@pytest.mark.parametrize("short,full,ok", [
    ("j r calloway", "james robert calloway", True),
    ("j calloway", "james robert calloway", True),
    ("k calloway", "james robert calloway", False),
    ("j r calloway", "james robert smith", False),
])
def test_initials(short, full, ok):
    assert initials_compatible(short, full) is ok


def test_initials_merge_into_the_titled_entity(resolved):
    assert eid(resolved, "J. R. Calloway") == eid(resolved, "James Robert Calloway")
    assert resolved.entities[eid(resolved, "J. R. Calloway")]["name"] == "James Robert Calloway"


def test_numbers_must_match(resolved):
    assert eid(resolved, "Apollo 11") != eid(resolved, "the Apollo 13")


def test_disambiguated_titles_stay_apart(resolved):
    assert eid(resolved, "Mercury (planet)") != eid(resolved, "Mercury (element)")


def test_embedding_rule_respects_the_numbers_guard():
    names = {"Apollo 11": [1, 0], "Apollo 13": [1, 0], "Robert Downey Jr": [0, 1], "Robert Downey Junior": [0, 1]}
    ex = {"Doc": {"entities": [ent(n, "OTHER") for n in names], "relations": []}}
    by_norm = {normalize(n): v for n, v in names.items()}
    fake_embed = lambda texts: np.array([by_norm.get(t, [0, 0]) for t in texts], dtype=float)  # title: no vector
    r = resolve(ex, fake_embed)
    assert eid(r, "Robert Downey Jr") == eid(r, "Robert Downey Junior")  # identical vectors, same numbers
    assert eid(r, "Apollo 11") != eid(r, "Apollo 13")                    # identical vectors, different numbers


# ---- graph ----

def test_expansion_reaches_the_bridge_paragraph(resolved):
    g = MemoryGraph(resolved)
    reached = [t for t, _ in g.expand([eid(resolved, "Night Harbor")], hops=2)]
    assert "James Robert Calloway" in reached and "Unrelated" not in reached


def test_linker_finds_entities_named_in_a_question(resolved):
    linker = MemoryGraph(resolved).linker()
    assert eid(resolved, "Night Harbor") in linker.link("Where was the director of Night Harbor born?")


@pytest.mark.parametrize("strategy", ["rerank", "slots"])
def test_graph_mode_adds_the_bridge_paragraph(resolved, monkeypatch, strategy):
    paras = {t: Paragraph(t, ("text",)) for t in EXTRACTIONS}
    monkeypatch.setattr(retrieve_mod, "GRAPH_STRATEGY", strategy)
    monkeypatch.setattr(retrieve_mod, "paragraphs", lambda: paras)
    monkeypatch.setattr(retrieve_mod.vectors, "search", lambda q, k: [("Night Harbor", 0.9), ("Unrelated", 0.5)])
    monkeypatch.setattr(retrieve_mod.vectors, "similarity", lambda q, titles: {t: 0.5 for t in titles})
    # fake reranker: prefers anything about the director
    monkeypatch.setattr(retrieve_mod, "rerank",
                        lambda q, titles, k: sorted(titles, key=lambda t: "Calloway" not in t)[:k])
    g = MemoryGraph(resolved)
    question = "Where was the director of Night Harbor born?"
    assert "James Robert Calloway" not in retrieve_mod.retrieve(question, "vector").titles
    graph = retrieve_mod.retrieve(question, "graph", g, g.linker())
    assert "James Robert Calloway" in graph.titles and len(graph.titles) <= retrieve_mod.FINAL_K
    assert ("James Robert Calloway", "BORN_IN", "Duluth, Minnesota") in g.triples(graph.titles)


# ---- metrics and schema ----

@pytest.mark.parametrize("pred,gold,em,f", [
    ("The Beatles", "beatles", True, 1.0), ("yes", "no", False, 0.0),
    ("Duluth", "Duluth, Minnesota", False, 2 / 3), ("", "Paris", False, 0.0),
])
def test_hotpot_metrics(pred, gold, em, f):
    assert exact_match(pred, gold) is em and f1(pred, gold) == pytest.approx(f)


def test_extraction_schema_rejects_invented_types():
    with pytest.raises(pydantic.ValidationError):
        Extraction.model_validate({"entities": [{"name": "X", "type": "SPACESHIP"}], "relations": []})
    with pytest.raises(pydantic.ValidationError):
        Extraction.model_validate({"entities": [], "relations": [{"source": "a", "relation": "LOVES", "target": "b"}]})
