import asyncio
import json
import logging
import math
import re
from pathlib import Path
from typing import Any

from llama_index.core import Settings

from src.agent.agent import answer
from src.config import configure_settings

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[1]
WITH_ANSWERS = _ROOT / "questions" / "dev_questions_with_answers.json"

SCALE = {"trillion": 1e12, "billion": 1e9, "million": 1e6, "thousand": 1e3}
# the sign can appear before or after the $ (e.g. "-5 million", "-$5", "$-5")
_NUM_RE = re.compile(
    r"(-?)\s*\$?\s*(-?)\s*([\d,]+(?:\.\d+)?)\s*(trillion|billion|million|thousand)?",
    re.IGNORECASE,
)


# pull every number-like token out of prose, scaling "billion"/"million" etc.
def extract_numbers(text: str) -> list[float]:
    numbers: list[float] = []
    for match in _NUM_RE.finditer(text):
        digits = match.group(3).replace(",", "")
        if not digits or digits == ".":
            continue
        value = float(digits)
        if "-" in (match.group(1), match.group(2)):
            value = -value
        scale = match.group(4)
        if scale:
            value *= SCALE[scale.lower()]
        numbers.append(value)
    return numbers


# numeric answer passes if ANY extracted number is close to gold. checking all numbers
# makes this robust to years / other figures in the prose. math.isclose uses abs_tol so
# a gold of 0 is not pathological (relative tolerance alone would require an exact 0).
def score_numeric(answer_text: str, gold: float, rel_tol: float = 0.01, abs_tol: float = 0.5) -> bool:
    return any(math.isclose(n, gold, rel_tol=rel_tol, abs_tol=abs_tol) for n in extract_numbers(answer_text))


JUDGE_PROMPT = (
    "You are grading a financial QA system. Decide whether the system answer is correct, "
    "meaning it captures the key facts of the gold answer. Respond with exactly one word: "
    "PASS or FAIL.\n\n"
    "Question: {question}\n"
    "Gold answer: {gold}\n"
    "System answer: {answer}\n"
)


def parse_verdict(text: str) -> bool:
    return "pass" in text.strip().lower()


# the three companies and the aliases the model might use for each
ENTITY_ALIASES = [
    {"apple", "aapl"},
    {"microsoft", "msft"},
    {"alphabet", "googl", "google"},
]


def _entities_in(text: str) -> set[int]:
    lowered = text.lower()
    return {i for i, aliases in enumerate(ENTITY_ALIASES) if any(a in lowered for a in aliases)}


# entity questions ("which company...") pass when the answer names the same company
# as the gold answer, allowing for aliases (Alphabet / GOOGL / Google).
def score_entity(answer_text: str, gold: str) -> bool:
    gold_entities = _entities_in(gold)
    return bool(gold_entities) and gold_entities <= _entities_in(answer_text)


async def score_with_llm(question: str, gold: str, answer_text: str, llm: Any) -> bool:
    prompt = JUDGE_PROMPT.format(question=question, gold=gold, answer=answer_text)
    resp = await llm.acomplete(prompt)
    return parse_verdict(str(resp))


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    configure_settings()
    items = json.loads(WITH_ANSWERS.read_text())
    passed = 0
    for q in items:
        resp = await answer(q["question"])
        if q["evaluation"] == "fuzzy_numeric":
            ok = score_numeric(resp.answer, q["gold_answer_numeric"])
        elif q["evaluation"] == "exact_match_entity":
            ok = score_entity(resp.answer, q["gold_answer"])
        else:
            ok = await score_with_llm(q["question"], q["gold_answer"], resp.answer, Settings.llm)
        passed += int(ok)
        logger.info("%s  %s  [%s]", "PASS" if ok else "FAIL", q["id"], q["evaluation"])
    logger.info("\n%d/%d correct (%.0f%%)", passed, len(items), 100 * passed / max(len(items), 1))


if __name__ == "__main__":
    asyncio.run(main())
