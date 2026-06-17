import pytest
from fastapi.testclient import TestClient

import src.api.backend as backend
from src.models import AgentResponse, SQLEvidence

client = TestClient(backend.app)


def test_chat_returns_agent_response(monkeypatch: pytest.MonkeyPatch) -> None:
    # mock the agent so the endpoint test hits no network
    fake = AgentResponse(
        answer="Apple revenue was $416.16B",
        sources_used=["sql"],
        evidence=[SQLEvidence(query="SELECT revenue FROM income_statements", rows=[{"revenue": 416161000000}])],
    )

    async def fake_answer(question: str) -> AgentResponse:
        return fake

    monkeypatch.setattr(backend, "answer", fake_answer)
    resp = client.post("/api/chat", json={"question": "apple revenue 2025"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "Apple revenue was $416.16B"
    assert body["sources_used"] == ["sql"]
    assert body["evidence"][0]["source"] == "sql"


def test_chat_passes_question_through(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, str] = {}

    async def fake_answer(question: str) -> AgentResponse:
        seen["q"] = question
        return AgentResponse(answer="ok", sources_used=[], evidence=[])

    monkeypatch.setattr(backend, "answer", fake_answer)
    client.post("/api/chat", json={"question": "what was revenue?"})
    assert seen["q"] == "what was revenue?"


def test_chat_missing_question_is_422() -> None:
    # required field missing -> FastAPI/pydantic validation error
    resp = client.post("/api/chat", json={})
    assert resp.status_code == 422


def test_chat_wrong_type_question_is_422() -> None:
    # a list is not a valid str, should be rejected before reaching the agent
    resp = client.post("/api/chat", json={"question": ["not", "a", "string"]})
    assert resp.status_code == 422
