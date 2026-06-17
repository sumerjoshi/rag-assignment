import asyncio
import sqlite3

import pytest

import eval.generate_dev_answers as gda
from eval.generate_custom_questions import build_questions
from eval.run_eval import extract_numbers, parse_verdict, score_entity, score_numeric, score_with_llm


# --- run_eval: number extraction + numeric scoring ---

def test_extract_numbers_handles_scales_and_commas() -> None:
    nums = extract_numbers("Revenue was $416.16 billion, up from 391,035,000,000.")
    assert 416160000000.0 in nums
    assert 391035000000.0 in nums


def test_score_numeric_ignores_years_in_prose() -> None:
    # "FY2025" must not cause a false match; the real figure should still pass
    assert score_numeric("Apple's FY2025 revenue was $416.16 billion", 416161000000) is True


def test_score_numeric_fails_when_value_wrong() -> None:
    assert score_numeric("Revenue was about $99 billion", 416161000000) is False


def test_score_numeric_matches_percentage() -> None:
    assert score_numeric("About 51.3% of revenue came from the US", 51.3) is True


def test_parse_verdict() -> None:
    assert parse_verdict("PASS") is True
    assert parse_verdict("fail") is False
    assert parse_verdict("FAIL, the numbers differ") is False


def test_score_entity_matches_company() -> None:
    gold = "Alphabet (GOOGL) had the fastest revenue growth at about 15.1%."
    assert score_entity("Alphabet (GOOGL) grew the fastest at 15.1%", gold) is True
    assert score_entity("Apple grew the fastest", gold) is False


def test_score_entity_accepts_alias() -> None:
    # the model may say "Google" where the gold says "Alphabet"
    gold = "Alphabet has the highest current ratio."
    assert score_entity("Google has the highest current ratio", gold) is True


def test_score_with_llm_uses_judge_verdict() -> None:
    # mock the llm so the judge makes no network call
    class _Resp:
        def __str__(self) -> str:
            return "PASS"

    class _LLM:
        async def acomplete(self, prompt: str) -> _Resp:
            return _Resp()

    ok = asyncio.run(score_with_llm("q", "gold", "answer", _LLM()))
    assert ok is True


# --- generate_custom_questions: gold comes from the DB oracle ---

def test_build_questions_uses_db_as_oracle() -> None:
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE income_statements "
        "(company_ticker TEXT, fiscal_year INT, period_type TEXT, "
        "revenue BIGINT, net_income BIGINT, gross_profit BIGINT)"
    )
    conn.execute(
        "INSERT INTO income_statements VALUES ('AAPL', 2025, 'FY', 416161000000, 112010000000, 195201000000)"
    )
    qs = build_questions(conn)
    rev = [q for q in qs if "total revenue" in q["question"] and "2025" in q["question"]]
    assert rev, "expected a revenue question for AAPL 2025"
    q = rev[0]
    assert q["gold_answer_numeric"] == 416161000000
    assert q["required_modalities"] == ["sql"]
    assert q["evaluation"] == "fuzzy_numeric"


# --- generate_dev_answers: output matches the {id: answer} format ---

def test_build_answers_produces_id_to_answer_map(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Resp:
        def __init__(self, ans: str) -> None:
            self.answer = ans

    async def fake_answer(question: str) -> _Resp:
        return _Resp(f"answer to: {question}")

    monkeypatch.setattr(gda, "answer", fake_answer)
    questions = [{"id": "q_001", "question": "Q1?"}, {"id": "q_006", "question": "Q2?"}]
    result = asyncio.run(gda.build_answers(questions))
    assert result == {"q_001": "answer to: Q1?", "q_006": "answer to: Q2?"}
