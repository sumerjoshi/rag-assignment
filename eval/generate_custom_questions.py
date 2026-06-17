import json
import logging
import sqlite3
from pathlib import Path

from src.config import ABSOLUTE_DB_PATH

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = _ROOT / "eval" / "custom_questions.json"

COMPANIES = {"AAPL": "Apple", "MSFT": "Microsoft", "GOOGL": "Alphabet"}
METRICS = {"revenue": "total revenue", "net_income": "net income", "gross_profit": "gross profit"}
YEARS = [2023, 2024, 2025]


def _fetch(conn: sqlite3.Connection, ticker: str, col: str, year: int) -> int | None:
    # col comes from the hardcoded METRICS dict, not user input, so interpolation is safe
    row = conn.execute(
        f"SELECT {col} FROM income_statements "
        "WHERE company_ticker = ? AND fiscal_year = ? AND period_type = 'FY'",
        (ticker, year),
    ).fetchone()
    return row[0] if row and row[0] is not None else None


# build a stress-test set whose gold answers come straight from the DB (the oracle)
def build_questions(conn: sqlite3.Connection) -> list[dict]:
    out: list[dict] = []
    qid = 1
    for ticker, name in COMPANIES.items():
        for col, label in METRICS.items():
            for year in YEARS:
                gold = _fetch(conn, ticker, col, year)
                if gold is None:
                    continue
                out.append(
                    {
                        "id": f"custom_{qid:03d}",
                        "question": f"What was {name}'s {label} in fiscal year {year}?",
                        "tier": 1,
                        "gold_answer": f"${gold:,}",
                        "gold_answer_numeric": gold,
                        "required_modalities": ["sql"],
                        "evaluation": "fuzzy_numeric",
                    }
                )
                qid += 1
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    conn = sqlite3.connect(f"file:{ABSOLUTE_DB_PATH}?mode=ro", uri=True)
    try:
        questions = build_questions(conn)
    finally:
        conn.close()
    OUTPUT.write_text(json.dumps(questions, indent=2))
    logger.info("generated %d questions -> %s", len(questions), OUTPUT)


if __name__ == "__main__":
    main()
