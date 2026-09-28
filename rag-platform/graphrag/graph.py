"""The knowledge graph: (:Entity)-[:REL {type, chunk}]->(:Entity) and (:Chunk)-[:MENTIONS]->(:Entity).

Neo4jGraph is the real store (python -m graphrag.graph loads it). MemoryGraph has the same interface and is used by
tests, so they don't need a database. Chunk ids are paragraph titles everywhere, so every answer can cite its source.
"""
import re
from collections import defaultdict, deque

from neo4j import GraphDatabase

from rag import config

from .resolve import Resolved, normalize

MAX_HOPS = 2
# Very common entities ("United States", "English") connect everything to everything; don't traverse through them.
HUB_DEGREE = 60


class Linker:
    """Finds graph entities named in a question by matching normalized names and aliases as whole words."""

    def __init__(self, names: dict[str, str]):  # surface name -> entity id
        self.by_norm = {}
        for name, eid in names.items():
            n = normalize(name)
            if len(n) >= 4 and not n.isdigit():
                self.by_norm.setdefault(n, eid)
        self.longest_first = sorted(self.by_norm, key=len, reverse=True)

    def link(self, question: str) -> list[str]:
        q = f" {normalize(question)} "
        found, taken = [], []
        for n in self.longest_first:
            for m in re.finditer(rf"(?<= ){re.escape(n)}(?= )", q):
                span = (m.start(), m.end())
                if not any(a < span[1] and span[0] < b for a, b in taken):  # skip names inside longer matches
                    taken.append(span)
                    found.append(self.by_norm[n])
        return list(dict.fromkeys(found))


class MemoryGraph:
    def __init__(self, resolved: Resolved):
        self.r = resolved
        self.adj = defaultdict(set)
        for src, _, dst, _ in resolved.relations:
            self.adj[src].add(dst)
            self.adj[dst].add(src)
        self.chunks_of = defaultdict(set)
        for chunk, ids in resolved.mentions.items():
            for eid in ids:
                self.chunks_of[eid].add(chunk)

    def linker(self) -> Linker:
        return Linker({n: e["id"] for e in self.r.entities.values() for n in [e["name"], *e["aliases"]]})

    def chunk_entities(self, chunks: list[str]) -> list[str]:
        return sorted({eid for c in chunks for eid in self.r.mentions.get(c, ())})

    def expand(self, seeds: list[str], hops: int = MAX_HOPS, limit: int = 20) -> list[tuple[str, int]]:
        paths = defaultdict(int)
        frontier = deque((s, 0) for s in seeds if s in self.r.entities)
        seen = set(seeds)
        while frontier:
            eid, depth = frontier.popleft()
            for chunk in self.chunks_of[eid]:
                paths[chunk] += 1
            if depth < hops and len(self.adj[eid]) <= HUB_DEGREE:
                for nxt in self.adj[eid] - seen:
                    seen.add(nxt)
                    frontier.append((nxt, depth + 1))
        return sorted(paths.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]

    def triples(self, chunks: list[str], limit: int = 15) -> list[tuple[str, str, str]]:
        wanted, names = set(chunks), {e["id"]: e["name"] for e in self.r.entities.values()}
        return [(names[s], t, names[d]) for s, t, d, c in self.r.relations if c in wanted][:limit]


class Neo4jGraph:
    def __init__(self):
        if not config.NEO4J_PASSWORD:
            raise RuntimeError("Set NEO4J_PASSWORD in .env and start Neo4j (docker compose up -d neo4j)")
        self.driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD))

    def close(self) -> None:
        self.driver.close()

    def run(self, query: str, **params) -> list[dict]:
        records, _, _ = self.driver.execute_query(query, **params)
        return [r.data() for r in records]

    def load(self, r: Resolved) -> None:
        self.run("MATCH (n) DETACH DELETE n")
        self.run("CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE")
        self.run("CREATE CONSTRAINT chunk_title IF NOT EXISTS FOR (c:Chunk) REQUIRE c.title IS UNIQUE")
        batch = lambda rows, n=1000: (rows[i:i + n] for i in range(0, len(rows), n))
        for rows in batch(list(r.entities.values())):
            self.run("UNWIND $rows AS row MERGE (e:Entity {id: row.id}) "
                     "SET e.name = row.name, e.type = row.type, e.aliases = row.aliases", rows=rows)
        mentions = [{"chunk": c, "entity": e, "about": r.subject[c] == e}
                    for c, ids in r.mentions.items() for e in ids]
        for rows in batch(mentions):
            self.run("UNWIND $rows AS row MERGE (c:Chunk {title: row.chunk}) WITH c, row "
                     "MATCH (e:Entity {id: row.entity}) MERGE (c)-[m:MENTIONS]->(e) SET m.about = row.about",
                     rows=rows)
        rels = [{"src": s, "type": t, "dst": d, "chunk": c} for s, t, d, c in r.relations]
        for rows in batch(rels):
            self.run("UNWIND $rows AS row MATCH (a:Entity {id: row.src}), (b:Entity {id: row.dst}) "
                     "MERGE (a)-[x:REL {type: row.type, chunk: row.chunk}]->(b)", rows=rows)

    def linker(self) -> Linker:
        rows = self.run("MATCH (e:Entity) RETURN e.id AS id, e.name AS name, e.aliases AS aliases")
        return Linker({n: row["id"] for row in rows for n in [row["name"], *(row["aliases"] or [])]})

    def chunk_entities(self, chunks: list[str]) -> list[str]:
        rows = self.run("MATCH (c:Chunk)-[:MENTIONS]->(e:Entity) WHERE c.title IN $chunks "
                        "RETURN DISTINCT e.id AS id", chunks=chunks)
        return sorted(r["id"] for r in rows)

    def expand(self, seeds: list[str], hops: int = MAX_HOPS, limit: int = 20) -> list[tuple[str, int]]:
        hops = max(0, min(int(hops), MAX_HOPS))  # interpolated into the pattern, so keep it a small int
        rows = self.run(
            f"""MATCH (e:Entity) WHERE e.id IN $seeds
                MATCH p = (e)-[:REL*0..{hops}]-(n:Entity)
                WHERE all(x IN nodes(p)[1..-1] WHERE COUNT {{ (x)--() }} <= $hub)
                MATCH (n)<-[:MENTIONS]-(c:Chunk)
                RETURN c.title AS title, count(*) AS paths
                ORDER BY paths DESC, title LIMIT $limit""",
            seeds=seeds, hub=HUB_DEGREE, limit=limit)
        return [(r["title"], r["paths"]) for r in rows]

    def triples(self, chunks: list[str], limit: int = 15) -> list[tuple[str, str, str]]:
        rows = self.run("MATCH (a:Entity)-[x:REL]->(b:Entity) WHERE x.chunk IN $chunks "
                        "RETURN a.name AS a, x.type AS t, b.name AS b LIMIT $limit", chunks=chunks, limit=limit)
        return [(r["a"], r["t"], r["b"]) for r in rows]


def build_resolved(embed: bool = True) -> Resolved:
    from rag.store import embedder

    from .extract import load
    extractions = load()
    if not extractions:
        raise SystemExit("No extractions yet. Run: python -m graphrag.extract")
    fn = (lambda texts: embedder().encode(texts, normalize_embeddings=True, batch_size=256)) if embed else None
    return resolve_and_report(extractions, fn)


def resolve_and_report(extractions: dict, embed_fn) -> Resolved:
    from .resolve import resolve
    r = resolve(extractions, embed_fn)
    print("Entity resolution:", r.stats)
    return r


if __name__ == "__main__":
    resolved = build_resolved()
    g = Neo4jGraph()
    g.load(resolved)
    counts = g.run("MATCH (e:Entity) WITH count(e) AS entities MATCH (c:Chunk) WITH entities, count(c) AS chunks "
                   "MATCH ()-[x:REL]->() RETURN entities, chunks, count(x) AS relations")
    print("Loaded into Neo4j:", counts[0])
    g.close()
