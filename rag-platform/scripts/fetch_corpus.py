"""Download ~100 Wikipedia articles on space exploration into data/corpus/.

Usage: python -m scripts.fetch_corpus
Articles are saved as plain text with trailing reference sections removed. Existing files are skipped,
so the corpus stays fixed once fetched (re-fetching can change the text and invalidate the eval set).
"""
import json
import re
import os
import time

import httpx

from rag import config

TITLES = [
    # Programs and crewed missions
    "Space Race", "Project Mercury", "Project Gemini", "Apollo program", "Apollo 8", "Apollo 11", "Apollo 13",
    "Apollo 17", "Skylab", "Space Shuttle program", "Space Shuttle Challenger disaster",
    "Space Shuttle Columbia disaster", "Artemis program", "Artemis I", "Commercial Crew Program",
    "Vostok 1", "Soyuz programme", "Soyuz (spacecraft)", "Salyut 1", "Mir", "International Space Station",
    "Tiangong space station", "Shenzhou 5",
    # People
    "Yuri Gagarin", "Valentina Tereshkova", "Neil Armstrong", "Buzz Aldrin", "Sally Ride", "Yang Liwei",
    "Wernher von Braun", "Sergei Korolev",
    # Early probes and landers
    "Sputnik 1", "Luna 2", "Luna 9", "Lunokhod 1", "Viking 1",
    # Telescopes
    "Hubble Space Telescope", "James Webb Space Telescope", "Spitzer Space Telescope",
    "Chandra X-ray Observatory", "Kepler space telescope", "Transiting Exoplanet Survey Satellite",
    "Gaia (spacecraft)",
    # Outer solar system
    "Voyager 1", "Voyager 2", "Pioneer 10", "Pioneer 11", "New Horizons", "Cassini–Huygens",
    "Galileo (spacecraft)", "Juno (spacecraft)",
    # Mars
    "Mars Pathfinder", "Sojourner (rover)", "Spirit (rover)", "Opportunity (rover)", "Curiosity (rover)",
    "Perseverance (rover)", "Ingenuity (helicopter)", "Mars Reconnaissance Orbiter", "Mars Express",
    "Mars Orbiter Mission", "Tianwen-1", "Zhurong (rover)",
    # Moon
    "Chang'e 4", "Chang'e 5", "Chandrayaan-2", "Chandrayaan-3",
    # Small bodies, inner planets and the Sun
    "Rosetta (spacecraft)", "Philae (spacecraft)", "Hayabusa2", "OSIRIS-REx", "Dawn (spacecraft)",
    "Double Asteroid Redirection Test", "MESSENGER", "BepiColombo", "Parker Solar Probe", "Solar Orbiter",
    # Vehicles
    "Saturn V", "Space Launch System", "Orion (spacecraft)", "SpaceX Dragon 2", "Falcon 9", "Falcon Heavy",
    "SpaceX Starship", "Energia (rocket)", "Buran (spacecraft)", "Ariane 5", "Ariane 6",
    "Long March (rocket family)",
    # Agencies, companies and sites
    "NASA", "European Space Agency", "Roscosmos", "Indian Space Research Organisation",
    "China National Space Administration", "JAXA", "Blue Origin", "Rocket Lab", "Kennedy Space Center",
    "Baikonur Cosmodrome",
    # Concepts and law
    "Outer Space Treaty", "Kármán line", "Geostationary orbit", "Lagrange point", "Gravity assist",
    "Space debris",
]

API = "https://en.wikipedia.org/w/api.php"
# Wikimedia rejects user agents without a contact URL or email. Set WIKIPEDIA_CONTACT to your own.
CONTACT = os.getenv("WIKIPEDIA_CONTACT", "personal learning project; https://www.mediawiki.org/wiki/API:Etiquette")
HEADERS = {"User-Agent": f"rag-platform-baseline/0.1 ({CONTACT})"}
# Sections from here to the end of the article are link lists and citations, not content.
TAIL_SECTIONS = re.compile(r"\n==\s*(See also|Notes|References|Citations|Sources|Bibliography|Further reading|"
                           r"External links|Footnotes)\s*==.*", re.S | re.I)


def slugify(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def fetch(client: httpx.Client, title: str) -> dict | None:
    r = client.get(API, params={"action": "query", "prop": "extracts|info", "explaintext": 1, "redirects": 1,
                                "inprop": "url", "titles": title, "format": "json", "formatversion": 2})
    r.raise_for_status()
    page = r.json()["query"]["pages"][0]
    if page.get("missing") or not page.get("extract"):
        return None
    text = TAIL_SECTIONS.sub("", page["extract"]).strip()
    return {"doc_id": slugify(title), "title": page["title"], "url": page["fullurl"], "text": text}


def main() -> None:
    config.CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    missing = []
    with httpx.Client(headers=HEADERS, timeout=30) as client:
        for title in TITLES:
            path = config.CORPUS_DIR / f"{slugify(title)}.json"
            if path.exists():
                continue
            doc = fetch(client, title)
            if doc is None:
                missing.append(title)
                continue
            (config.CORPUS_DIR / f"{doc['doc_id']}.json").write_text(
                json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"saved {doc['title']} ({len(doc['text']):,} chars)")
            time.sleep(0.2)
    total = len(list(config.CORPUS_DIR.glob("*.json")))
    print(f"\n{total} documents in {config.CORPUS_DIR}")
    if missing:
        print("Not found:", ", ".join(missing))


if __name__ == "__main__":
    main()
