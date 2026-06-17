import asyncio
import json
from pathlib import Path

from src.agent.agent import answer

_ROOT = Path(__file__).resolve().parents[1]
DEV_QUESTIONS = _ROOT / "questions" / "dev_questions.json"
OUTPUT = _ROOT / "dev_answers.json"


# build the {id: answer} mapping that matches dev_answers_example.json
async def build_answers(questions: list[dict]) -> dict[str, str]:
    results: dict[str, str] = {}
    for q in questions:
        resp = await answer(q["question"])
        results[q["id"]] = resp.answer
    return results


async def main() -> None:
    questions = json.loads(DEV_QUESTIONS.read_text())
    results = await build_answers(questions)
    OUTPUT.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
