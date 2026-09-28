"""Test fixtures. Every test session gets its own throwaway vector store and audit database, built from the
seed manifest, so tests never touch data/ and run the same in CI."""
import pytest
from fastapi.testclient import TestClient

from eval import leakage_cases
from rag import config, pipeline, store, tenant_ingest
from rag.acl import User
from rag.auth import issue_token


@pytest.fixture(scope="session", autouse=True)
def isolated_store(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("store")
    mp = pytest.MonkeyPatch()
    mp.setattr(config, "QDRANT_URL", "")
    mp.setattr(config, "QDRANT_PATH", tmp / "qdrant")
    mp.setattr(config, "AUDIT_DB", tmp / "audit.db")
    mp.setattr(config, "JWT_SECRET", "test-secret-" + "x" * 40)
    store.qdrant.cache_clear()
    tenant_ingest.ingest(include_filler=False)
    yield
    store.qdrant().close()
    store.qdrant.cache_clear()
    mp.undo()


@pytest.fixture(scope="session")
def manifest() -> dict:
    return leakage_cases.load_manifest()


@pytest.fixture(scope="session")
def users(manifest) -> dict[str, User]:
    return {u.id: u for u in leakage_cases.users(manifest)}


@pytest.fixture
def fake_llm(monkeypatch):
    """Replace the Claude call so API tests are free and deterministic. Records what the model would see."""
    calls = []

    def generate(question, hits, model=config.ANSWER_MODEL):
        calls.append({"question": question, "hits": hits})
        return "stub answer [1]", "end_turn", 0, 0

    monkeypatch.setattr(pipeline, "generate", generate)
    return calls


@pytest.fixture(scope="session")
def client():
    from rag.api import app
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def auth(users):
    def headers(user_id: str) -> dict:
        return {"Authorization": f"Bearer {issue_token(users[user_id])}"}
    return headers
