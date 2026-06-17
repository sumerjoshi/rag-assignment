import asyncio
import json
import logging
from pathlib import Path

from src.agent.agent import answer

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[1]
DEV_QUESTIONS = _ROOT / "questions" / "dev_questions.json"
OUTPUT = _ROOT / "dev_answers.json"


# build the {id: answer} mapping that matches dev_answers_example.json
async def build_answers(questions: list[dict]) -> dict[str, str]:
    results: dict[str, str] = {}
    for q in questions:
        # one failing question (API error/timeout) should not lose the whole run
        try:
            resp = await answer(q["question"])
            results[q["id"]] = resp.answer
        except Exception as exc:
            logger.warning("question %s failed: %s", q["id"], exc)
            results[q["id"]] = f"[error generating answer: {exc}]"
    return results


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    questions = json.loads(DEV_QUESTIONS.read_text())
    results = await build_answers(questions)
    OUTPUT.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
