"""Entity resolution: collapse the many surface forms of one entity into a single node.

A graph full of duplicates can't connect anything: if one paragraph says "J. Smith" and another "John Smith",
traversal never crosses between them. Three rules, applied with union-find:

1. Same normalized name (case, accents, punctuation, a leading "the", and a trailing Wikipedia-style
   "(film)" are ignored). Disambiguated titles that share a base name ("Mercury (planet)", "Mercury (element)")
   are never merged with each other, and a bare mention links to one of them only if it's unambiguous.
2. Person initials: "J. R. Smith" joins "John Robert Smith" when exactly one full name is compatible.
3. Embedding similarity of the names >= EMBED_THRESHOLD, only for names that share a word and have identical
   numbers ("Apollo 11" must never merge with "Apollo 13").

Paragraph titles are Wikipedia article names, so when a group contains one it becomes the canonical name.
"""
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import numpy as np

EMBED_THRESHOLD = 0.93
_PAREN = re.compile(r"\s*\([^)]*\)\s*$")
_NON_WORD = re.compile(r"[^\w\s]")
_DIGITS = re.compile(r"\d+")


def normalize(name: str, keep_paren: bool = False) -> str:
    s = unicodedata.normalize("NFKD", name)
    s = "".join(c for c in s if not unicodedata.combining(c)).lower().strip()
    if not keep_paren:
        s = _PAREN.sub("", s)
    s = _NON_WORD.sub(" ", s)
    s = " ".join(s.split())
    return s[4:] if s.startswith("the ") else s


def digits(name: str) -> tuple[str, ...]:
    return tuple(_DIGITS.findall(name))


def initials_compatible(short: str, full: str) -> bool:
    """'j r smith' vs 'john robert smith': same last name, and each initial matches a first-name word in order."""
    s, f = short.split(), full.split()
    if len(s) < 2 or len(f) < 2 or s[-1] != f[-1] or s == f:
        return False
    firsts, candidates = s[:-1], f[:-1]
    if not all(len(t) == 1 for t in firsts):
        return False
    it = iter(candidates)
    return all(any(c.startswith(t) for c in it) for t in firsts)


class UnionFind:
    def __init__(self):
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra
            return True
        return False


@dataclass
class Resolved:
    entities: dict[str, dict] = field(default_factory=dict)         # id -> {name, type, aliases}
    mentions: dict[str, set[str]] = field(default_factory=dict)      # chunk title -> entity ids
    relations: list[tuple[str, str, str, str]] = field(default_factory=list)  # (src, type, dst, chunk)
    subject: dict[str, str] = field(default_factory=dict)            # chunk title -> subject entity id
    stats: dict[str, int] = field(default_factory=dict)


def resolve(extractions: dict[str, dict], embed=None) -> Resolved:
    """extractions: chunk title -> {"entities": [{name, type}], "relations": [{source, relation, target}]}.
    embed: optional callable(list[str]) -> np.ndarray of normalized vectors, for rule 3."""
    names: Counter = Counter()          # surface name -> mentions
    types: dict[str, Counter] = defaultdict(Counter)
    titles = set(extractions)
    for title, ex in extractions.items():
        names[title] += 1
        for e in ex["entities"]:
            names[e["name"]] += 1
            types[e["name"]][e["type"]] += 1

    # Group keys: titles keep their disambiguating parenthetical; everything else uses the base form.
    base_titles: dict[str, list[str]] = defaultdict(list)
    for t in titles:
        base_titles[normalize(t)].append(t)

    def key_of(name: str) -> str:
        if name in titles:
            return "title:" + normalize(name, keep_paren=True)
        base = normalize(name)
        owners = base_titles.get(base, [])
        return "title:" + normalize(owners[0], keep_paren=True) if len(owners) == 1 else "name:" + base

    uf = UnionFind()
    keys = {n: key_of(n) for n in names if normalize(n)}
    for k in set(keys.values()):
        uf.find(k)
    stats = Counter(surface_names=len(keys), groups_by_normalization=len(set(keys.values())))

    # Rule 2: person initials
    person_keys = {keys[n]: normalize(n) for n in keys if types[n].most_common(1) and
                   types[n].most_common(1)[0][0] == "PERSON"}
    fulls_by_last = defaultdict(list)
    for k, norm in person_keys.items():
        if not all(len(t) == 1 for t in norm.split()[:-1]):
            fulls_by_last[norm.split()[-1] if norm else ""].append((k, norm))
    for k, norm in person_keys.items():
        if norm and len(norm.split()) >= 2 and all(len(t) == 1 for t in norm.split()[:-1]):
            matches = [fk for fk, fn in fulls_by_last[norm.split()[-1]] if initials_compatible(norm, fn)]
            if len(set(matches)) == 1 and uf.union(matches[0], k):
                stats["merged_by_initials"] += 1

    # Rule 3: embedding similarity between groups that share a word and have identical numbers
    if embed is not None:
        reps = {}
        for n, k in keys.items():
            reps.setdefault(uf.find(k), n)
        group_ids = list(reps)
        texts = [normalize(reps[g]) for g in group_ids]
        vecs = embed(texts)
        by_word = defaultdict(list)
        for i, t in enumerate(texts):
            for w in set(t.split()):
                if len(w) > 2:
                    by_word[w].append(i)
        checked = set()
        for idx in by_word.values():
            if len(idx) > 200:  # very common words ("john", "river") carry no signal
                continue
            for a in range(len(idx)):
                for b in range(a + 1, len(idx)):
                    i, j = idx[a], idx[b]
                    if (i, j) in checked:
                        continue
                    checked.add((i, j))
                    gi, gj = group_ids[i], group_ids[j]
                    if gi.startswith("title:") and gj.startswith("title:"):
                        continue  # two Wikipedia articles are two entities by construction
                    if digits(texts[i]) != digits(texts[j]):
                        continue
                    if float(np.dot(vecs[i], vecs[j])) >= EMBED_THRESHOLD and uf.union(gi, gj):
                        stats["merged_by_embedding"] += 1

    # Build canonical entities
    members = defaultdict(list)
    for n, k in keys.items():
        members[uf.find(k)].append(n)
    out = Resolved()
    name_to_id = {}
    for root, group in members.items():
        title_names = [n for n in group if n in titles]
        canonical = title_names[0] if title_names else max(group, key=lambda n: (names[n], len(n)))
        type_votes = sum((types[n] for n in group), Counter())
        eid = "e:" + normalize(canonical, keep_paren=True)
        out.entities[eid] = {"id": eid, "name": canonical,
                             "type": type_votes.most_common(1)[0][0] if type_votes else "OTHER",
                             "aliases": sorted({n for n in group if n != canonical})}
        for n in group:
            name_to_id[n] = eid
    stats["entities"] = len(out.entities)

    for title, ex in extractions.items():
        ids = {name_to_id[e["name"]] for e in ex["entities"] if e["name"] in name_to_id}
        out.subject[title] = name_to_id[title]
        out.mentions[title] = ids | {out.subject[title]}
        for r in ex["relations"]:
            src, dst = name_to_id.get(r["source"]), name_to_id.get(r["target"])
            if src and dst and src != dst:
                out.relations.append((src, r["relation"], dst, title))
            else:
                stats["relations_dropped_unknown_entity"] += 1
    stats["relations"] = len(out.relations)
    out.stats = dict(stats)
    return out
