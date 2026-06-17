# Fireworks AI AMLE Take-Home: Agentic RAG for 10-K Analysis

This take-home is meant to mirror part of the Applied Machine Learning Engineer role: supporting customers in their journey to build GenAI applications on Fireworks.

In this exercise, you should approach the problem like a Fireworks engineer supporting a customer who needs an agentic RAG workflow over structured financial data and 10-K filings.

## What We're Looking For

1. Customer-oriented problem solving: translate the customer's needs into a practical system design.
2. Agent and tool design: decide when to query SQL, search PDFs, or combine both.
3. Evaluation discipline: show how you measured quality and where the system still fails.
4. Practical trade-offs: explain choices around models, latency, cost, reliability, and complexity.
5. Communication: provide clear instructions, clear answers, and a concise technical report.

## Customer Scenario

**From:** Natalie Brooks <natalie.brooks@acmecorp.example.com>  
**To:** Solutions Team <solutions@fireworks.ai>  
**Subject:** Help Needed: Local Research Assistant for 10-K Analysis

Hi Fireworks team,

Our research team spends a lot of time reading annual reports, cross-checking management commentary against financial tables, and building simple comparisons across companies. We have a local dataset that combines structured financial data with the original 10-K filings, and we want a local AI assistant that can help analysts answer increasingly complex questions over that material.

Our current prototype can handle simple lookups, but it breaks down when a question requires planning, multiple retrieval steps, or combining narrative disclosures with structured financials. In particular, we need a system that can:

- decide when to query the SQLite database versus the filings
- gather evidence from the right sources
- answer questions that range from direct lookup to multi-step synthesis
- stay grounded in the provided documents and data

We are providing:

- six 10-K filings for Apple, Microsoft, and Alphabet across FY2024 and FY2025
- a local SQLite database with structured financial data
- a 10-question development set

We would like a local proof of concept that a reviewer can run on their machine and interact with directly.

Thanks,  
Natalie Brooks  
Director of Research Systems, Acme Corp

## Project Structure

- `data/`: generated or provided assignment data, including the SQLite DB and 10-K PDFs
- `questions/`: the development-set questions, public dev answer key, and the `dev_answers.json` example template
- `scripts/`: helper scripts that can fetch the SEC source data, render PDFs, and build `financials.db`
- `starter/`: lightweight starter dependencies for setup and experimentation
- `setup.sh`: end-to-end local setup script

## What You Receive

- `questions/dev_questions.json`: 10 development-set questions
- `questions/dev_questions_with_answers.json`: the public dev-set answer key
- `questions/dev_answers_example.json`: template for your `dev_answers.json`
- `setup.sh`: local bootstrap script
- `starter/requirements.txt`: setup and starter dependencies
- `scripts/`: scripts that can fetch or rebuild the data if it is not already present

If `data/financials.db` and the 10-K PDFs are already present, `setup.sh` will reuse them. It only fetches SEC data if it needs to rebuild missing assets.

The dev-set answer key is public so you can evaluate your system locally. We intentionally do not provide an evaluation harness; part of the assignment is deciding how to measure correctness against the provided questions, answers, and data. Fireworks keeps a separate held-out set for the hidden final evaluation.

## Data Overview

The SQLite database includes these tables:

- `companies`: company metadata
- `income_statements`: revenue, gross profit, operating income, net income, EPS, and R&D
- `balance_sheets`: assets, liabilities, equity, cash, debt, and current balance metrics
- `segment_revenue`: revenue by business segment
- `geographic_revenue`: revenue by geography

The filings provide the narrative context needed for questions about risks, strategy, segment definitions, geographic commentary, and management discussion.

## Your Task

Build a local agentic RAG system that can answer increasingly complex questions about the provided companies and filings.

Your system should:

- run locally on a reviewer's machine
- support interactive use (e.g., with a simple UI)
- expose an HTTP API at `http://localhost:8000/api/chat` that accepts `POST` requests with `{"question": "..."}` and returns the answer either as JSON with a top-level `answer` or `content` field (for example, `{"answer": "..."}`) or as an SSE stream with an `answer` event whose data is `{"content": "..."}`.
- route questions to the right source or sources
- return grounded answers that make it easy to inspect evidence
- handle both straightforward retrieval and multi-step reasoning

## Submission Guidelines

- Submit within the deadline provided by your recruiter.
- You may use any Fireworks model and additional framework, database, or vector store.
- You may use the internet, documentation, third-party packages, and AI coding tools.
- If you use AI assistance, mention how in your report.
- Keep external API usage to a reasonable prototype budget.

## Required Deliverables

- A zip file containing your implementation.
- A `README` in your submission with exact local run instructions, required environment variables, and any setup steps.
- A local interactive entry point so a reviewer can ask ad hoc questions.
- A `dev_answers.json` file with your answers to the 10 development questions.
- A short report, about 1 to 2 pages, covering:
  - what you built
  - how the system is structured
  - how you retrieve from SQL and PDFs
  - how you evaluate the system
  - what trade-offs you made and why
  - what you would improve with more time

## `dev_answers.json` Format

Create `dev_answers.json` by copying `questions/dev_answers_example.json`, then fill in your answers as a JSON object keyed by question ID:

```json
{
  "q_001": "<your answer>",
  "q_006": "<your answer>",
  "q_008": "<your answer>"
}
```

Answers may be short or long depending on the question. For synthesis questions, concise but well-supported answers are preferred.

Because the dev answer key is public, `dev_answers.json` is not the hidden evaluation target. We still ask you to submit it so we can see the exact outputs your final system produced on the public development set.

## Getting Started

Run:

```bash
./setup.sh
```

What `setup.sh` does:

- creates a local virtual environment with `uv`
- installs setup and starter dependencies
- downloads the SEC companyfacts JSON if needed
- renders the six 10-K PDFs if needed
- builds `data/financials.db` if needed

Then inspect:

- `data/financials.db`
- `data/pdfs/`
- `questions/dev_questions.json`
- `questions/dev_questions_with_answers.json`

You should use the public answer key to design your own evaluation approach for the dev set.

## How We Will Review

We will review your submission using:

- the quality of the local interactive system
- your ability to route between SQL and PDF-based evidence
- how thoughtfully you evaluate your system against the public dev set
- the clarity of your report and trade-off discussion
- an internal held-out evaluation set

---

# Solution

Everything above this line is the original take-home brief. Everything below is my
implementation: what it is, how to set it up, and how to run it.

The system is a local agentic RAG app over the provided data. A single LLM agent answers
questions about Apple, Microsoft, and Alphabet by routing to two tools: a text-to-SQL
engine over `financials.db` for exact numbers, and vector search over the 10-K filings for
narrative context. It can use either tool or both, and every answer comes back with the
evidence that supports it.

## How it works

For a single question:

1. The request reaches the agent (directly, through the HTTP API, or through the UI).
2. The agent (LlamaIndex `FunctionAgent`) decides which tool(s) to call based on the question.
   - `query_financials` runs text-to-SQL against the SQLite database and returns rows.
   - `search_filings` runs vector search over the embedded 10-K chunks and returns passages.
3. Each tool call records structured evidence (the SQL query and rows, or the PDF passages
   with company, fiscal year, page, and similarity score).
4. The agent writes a final answer grounded in the tool outputs and returns an
   `AgentResponse` with the answer, the sources it used, and the evidence list.

Numeric and structured questions go to SQL, narrative questions go to the filings, and
multi-step questions (for example "how did revenue change and why") use both.

## Repository layout

The code I added lives under `src/`, `eval/`, and `tests/`:

```
src/
  config.py              env loading, Fireworks LLM/embedding clients, resolved paths
  models.py              Pydantic models (SQLEvidence, PDFEvidence, AgentResponse)
  retrieval/sql.py       text-to-SQL engine, schema context string, SQL evidence
  retrieval/pdf.py       vector index load (lazy build), PDF retrieval, PDF evidence
  ingest/build_index.py  PDF to chunks to embeddings to persisted vector store
  agent/agent.py         the routing agent and the answer() entry point
  api/backend.py         FastAPI app exposing POST /api/chat
  ui/ui.py               Streamlit interface
eval/
  generate_dev_answers.py       produces dev_answers.json
  run_eval.py                   scores the agent against the public answer key
  generate_custom_questions.py  generates an extra SQL stress set from the database
scripts/build_index.sh          interactive wrapper around the ingestion step
tests/                          test suite (pytest), mypy-clean
```

## Prerequisites

- Python 3.13
- `uv`, used by `setup.sh` (https://github.com/astral-sh/uv)
- A Fireworks API key

## Setup

1. Run the bootstrap script. It creates a `.venv`, installs `src/requirements.txt`, and
   builds `data/financials.db` and the six PDFs if they are missing.

```bash
./setup.sh
source .venv/bin/activate
```

2. Create your environment file and set your Fireworks key.

```bash
cp .env.example .env
# open .env and set FIREWORKS_API_KEY to your real key
```

### How environment loading works

`src/config.py` calls `load_dotenv()` (from `python-dotenv`) when it is imported, which
reads `.env` into the process environment. Two things worth knowing:

- The Fireworks values are validated lazily, at the point where the LLM or embedding client
  is actually built, not at import time. Importing a module or running path-only code does
  not require a key, so the test suite and tooling run without one.
- The path values have sensible defaults relative to the repo root, so they are optional.
  Set them only if you want to override the defaults.

`.env` is gitignored, so real keys are never committed.

### Environment variables

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `FIREWORKS_API_KEY` | yes | none | Fireworks API key |
| `FIREWORKS_BASE_URL` | yes | set in `.env.example` | OpenAI-compatible Fireworks base URL |
| `FIREWORKS_LLM_MODEL` | yes | set in `.env.example` | agent and text-to-SQL model |
| `FIREWORKS_EMBEDDING_MODEL` | yes | set in `.env.example` | embedding model for the PDF index. Must match at build and query time |
| `FIREWORKS_RERANK_MODEL` | no | set in `.env.example` | reserved for a future reranker, not used yet |
| `DATABASE_PATH` | no | `data/financials.db` | SQLite database path |
| `PDF_DIR` | no | `data/pdfs` | folder of 10-K PDFs |
| `VECTOR_STORE_DIR` | no | `storage` | where the embedded index is persisted |
| `API_HOST`, `API_PORT` | no | `0.0.0.0`, `8000` | present for reference. The server listens on `localhost:8000` to match the required API URL |

## Build the vector store

`setup.sh` does not build the PDF index, because that step calls the Fireworks embedding
API. Build it once with either command:

```bash
./scripts/build_index.sh            # interactive, prompts before building
python -m src.ingest.build_index    # direct, pass --force to rebuild from scratch
```

This reads the PDFs, splits them into chunks, embeds them with the Fireworks embedding
model, and persists the index to `storage/`. If you skip this step the index builds
automatically the first time you ask a question, so the first request is just slower.

## Run the system

Activate the venv first (`source .venv/bin/activate`), then start either entry point.

### HTTP API

```bash
python -m src.api.backend
```

This serves `POST http://localhost:8000/api/chat`. It accepts `{"question": "..."}` and
returns the answer plus its evidence:

```bash
curl -s -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What was Apple total revenue in fiscal year 2025?"}'
```

Response shape:

```json
{
  "answer": "Apple's total revenue for fiscal year 2025 was $416.161 billion.",
  "sources_used": ["sql"],
  "evidence": [
    {
      "source": "sql",
      "query": "SELECT revenue FROM income_statements WHERE company_ticker = 'AAPL' AND fiscal_year = 2025",
      "rows": [{"revenue": 416161000000}]
    }
  ]
}
```

### Interactive UI

```bash
streamlit run src/ui/ui.py
```

This opens a local Streamlit app (default `http://localhost:8501`) with a query box. It
shows the answer, which sources were used, and an expandable evidence panel with the SQL
rows and the PDF passages.

## Tests

```bash
pytest -v tests/
mypy src tests
```

The suite runs without network access. The agent, LLM, and embedding calls are mocked, so
it is fast and deterministic. It covers config and path handling, the Pydantic models, the
SQL and PDF evidence mapping, ingestion, agent routing and evidence collection, the API
contract, and the eval scoring logic.

## Evaluation

```bash
python -m eval.generate_dev_answers       # writes dev_answers.json (the required deliverable)
python -m eval.run_eval                   # scores the agent against the public answer key
python -m eval.generate_custom_questions  # writes eval/custom_questions.json, an extra SQL stress set
```

`run_eval` uses two scorers, matching the `evaluation` field in the public answer key: a
fuzzy numeric match for figures (extract the number from the answer and compare it to the
gold value within a tolerance) and an LLM judge for narrative answers.
`generate_custom_questions` builds extra questions whose gold answers come straight from the
database, so they are exact by construction and need no manual labeling.

## Deliverables

- `dev_answers.json` at the repo root: my answers to the 10 development questions.
- The short report covering design, retrieval, evaluation, trade-offs, and AI assistance is
  in `REPORT.pdf`.
