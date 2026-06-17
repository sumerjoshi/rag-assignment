
from sqlalchemy import create_engine
from src.config import ABSOLUTE_DB_PATH
from llama_index.core import SQLDatabase
from llama_index.core.query_engine import NLSQLTableQueryEngine

TABLES = ["companies", "income_statements", "balance_sheets", "segment_revenue", "geographic_revenue"]

# Text-to-SQL: NLSQLTableQueryEngine sends the user's natural-language question plus
# the CONTEXT_SCHEMA_STR below to the LLM, which generates a read-only SELECT and runs
# it against financials.db. The context string grounds the LLM in the real schema,
# value encodings, and categorical domains so the generated SQL is correct (not hallucinated).
#
# The context string was hand-written from inspecting the DB directly (all read-only):
#   sqlite3 "file:data/financials.db?mode=ro" ".tables"
#   sqlite3 "file:data/financials.db?mode=ro" ".schema"
#   sqlite3 -header -column "file:data/financials.db?mode=ro" "SELECT * FROM companies LIMIT 10;"
#   sqlite3 -header -column "file:data/financials.db?mode=ro" \
#     "SELECT company_ticker, fiscal_year, period_type, COUNT(*) FROM income_statements GROUP BY 1,2,3;"
#   sqlite3 -header -column "file:data/financials.db?mode=ro" \
#     "SELECT company_ticker, GROUP_CONCAT(DISTINCT segment_name) FROM segment_revenue GROUP BY 1;"
#   sqlite3 -header -column "file:data/financials.db?mode=ro" "SELECT DISTINCT region FROM geographic_revenue;"
#   sqlite3 -header -line   "file:data/financials.db?mode=ro" \
#     "SELECT * FROM income_statements WHERE company_ticker='AAPL' AND period_type='FY' LIMIT 1;"
# Key findings: monetary values are raw US dollars; annual rows use period_type='FY';
# tickers are AAPL/MSFT/GOOGL across fiscal years 2023-2025.

# Given this information from above, defining Context Schema Below
CONTEXT_SCHEMA_STR = """
This SQLite database has annual financial data pulled from 10-K filings for three
companies: AAPL (Apple Inc.), MSFT (Microsoft Corporation), and GOOGL (Alphabet Inc.).
Fiscal years 2023 through 2025 are available. Only generate read-only SELECT queries.

A few things to keep in mind when writing queries:
- Money columns are in raw US dollars, so 416161000000 means about 416 billion dollars.
  Don't assume the numbers are in thousands or millions.
- Annual rows have period_type = 'FY'. There is no 'annual' or 'year' value, so always
  filter on period_type = 'FY' when you want full year numbers.
- Join the fact tables back to companies on company_ticker = companies.ticker.
- Fiscal year end is different per company (AAPL in September, MSFT in June, GOOGL in December).

Tables:
- companies(ticker, name, cik, sic, sector, fiscal_year_end)
- income_statements(company_ticker, fiscal_year, period_start, period_end, period_type,
    revenue, cost_of_revenue, gross_profit, research_and_development,
    total_operating_expenses, operating_income, net_income, eps_basic, eps_diluted)
- balance_sheets(company_ticker, fiscal_year, period_end, period_type, total_assets,
    total_liabilities, stockholders_equity, cash_and_equivalents, total_debt,
    short_term_debt, accounts_receivable, total_current_assets, total_current_liabilities)
- segment_revenue(company_ticker, fiscal_year, period_end, period_type, segment_name, revenue)
    segment_name values depend on the company, for example AAPL uses 'iPhone', 'Mac', 'iPad',
    'Services', 'Wearables, Home and Accessories'; MSFT uses 'Intelligent Cloud',
    'Productivity and Business Processes', 'More Personal Computing'; GOOGL uses
    'Google Services', 'Google Cloud', 'Other Bets'.
- geographic_revenue(company_ticker, fiscal_year, period_end, period_type, region, revenue)
    region values also depend on the company, for example 'Americas', 'EMEA', 'APAC',
    'United States', 'Greater China', 'Europe', 'Japan', 'Rest of Asia Pacific'.
""".strip()


def build_sql_query_engine() -> NLSQLTableQueryEngine:
    """Reading the SQL Database as NLSQLTableQueryEngine so we can run Text-to-SQL over it"""
    engine = create_engine(f"sqlite:///file:{ABSOLUTE_DB_PATH}?mode=ro&uri=true")
    sql_db = SQLDatabase(engine, include_tables=TABLES)
    return NLSQLTableQueryEngine(sql_database=sql_db, context_str_prefix=CONTEXT_SCHEMA_STR) 

