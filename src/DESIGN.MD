## Scratch Work DESIGN

## What I'm being Judged against

- the quality of the local interactive system
- your ability to route between SQL and PDF-based evidence
- how thoughtfully you evaluate your system against the public dev set
- the clarity of your report and trade-off discussion
- an internal held-out evaluation set



Inputs
- setup.sh (script that sets up the environment and basically has a checklist of what to do)
- questions folder that is basically an eval harness
- how we are able to run the scripts here and set up the data
- and then an eval set

**Constraints**
- run locally on a reviewer's machine
- support interactive use (e.g., with a simple UI)
- expose an HTTP API at `http://localhost:8000/api/chat` that accepts `POST` requests with `{"question": "..."}` and returns the answer either as JSON with a top-level `answer` or `content` field (for example, `{"answer": "..."}`) or as an SSE stream with an `answer` event whose data is `{"content": "..."}`.
- route questions to the right source or sources
- return grounded answers that make it easy to inspect evidence
- handle both straightforward retrieval and multi-step reasoning

**Deliverables**
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

**Rewritten Problem Statement**

After taking a look at dev_answers_example.json, it seems that a customer wants to be able to write natural language queries or criterion over data basically, and be able to get data back to the user. It seem that each criterion should be marked as 
    - hard (exact match or must match)
    - soft (nice to have)

And basically get back an answer based on that modality.

Since the problem is based financial documents, we 6 companies
with financial data basically listed here in the data folder 
with an actual financials.db database. 

**High Level Design**

**Functional Requirements**
- Customer can be able to write or mark a list of natural language ``required_modalities``
    required_modalities -> sql, pdf, or both
``question`` -> which is just a text string
- Searching against 6 financial documents
- Returning the best answer given the requirements back to the user


**Non Functional Requirements**
- Low Latency: For the take home, we want to make sure that the responsiveness seems fast
- Available: The Fireworks AI Team should be able to bring up the servers they need to bring up and it should run
- Consistency - this data is not going to change currently

**Core Entities**
What I don't need to design.
- Financial Tables (companies, income_statements, balance_sheets, segment_revenue, geographic revenue)

What I do need to Design:
We need to have the data that is coming back from the Vector DB

ResultChunk
    id
    company_ticker
    fiscal_year
    page_number
    chunk_id (may need this)
    section (if we can reliably get this somehow)
    text
    score (this would be return from the vector DB)
    embedding (this would be the insert only)

Pydantic Modeling of the Responses
Basically when we are returning data back to the user, we need to validate the data. So we would use Pydantic V2 here.

(Pydantic Notation)
SqlEvidence 
    source (Literal) -> sql
    query: str
    rows: list[dict]
    # can include the table it comes from

(Pydantic Notation)
Pdf Evidence
    source (Literal) -> pdf
    chunk_id: str
    company_ticket: str 
    # maybe include company_name here
    fiscal_year
    page_number
    # section?
    text: str
    score: double/float

We want to combine these two pydantic models into
a discriminated Union here so we can basically give the
same model response back to the user
    #Evidence = would use the Annotated + [x | y] functionality here

Then finally, we want to give the AgentResponse

AgentResponse:
    answer: str
    sources_used: (can only be sql and or pdf)
    evidence (Evidence model as a list)

**How would I translate the natural language into a sql query**
We would need to find a text to sql tool that is able 
to convert natural language text into sql and then basically
run a read only query on the sql table and give me a response.

SQL Output
After looking on line, LlamaIndex seems to be a quick option using NLSqlTableQueryEngine (a library within LlamaIndex) with the sqllite connector.
    -> pass in context_str to understand the context
    -> make it read only
    -> when we go to the database here, we output the raw rows and we can put it as the Pydantic Model

PDF Output
    -> have to use the right Query Engine Tool here basically but 
    same thing here.
    -> Declare MetadataFilters during query time i believe

**Agent Ties it Together**
Since the document asks for multi-step reasoning and straightforward reasoning, we can stick to using LlamaIndex
QueryEngineTool (for both the SQL Engine and PDF Engine) and then have to use AgentWorkflow library to tie this together

**Vector DB**
I think here we would want to use (because this is a POC) either Chroma or LlamaIndex's VectorStoreIndex. We load from disk here when we launch the application and then embed on setup or on upsert.

**Write Path or Ingestion**
The rough outline of the write pipeline would be to take all the pdfs, chunk them, attach any metadata -> embedding and then persistence
    -> take pdf files using pymupdf
    -> split the text from the documents in chunks
    -> embed the chunks to generate embeddings
    -> upsert that to our vector_db
-> we can write a separate setup script in our starter folder to do this
-> For the SQL Data or Side, don't need to do this because we have financials db already created for us.

**Read Path at Query Time**
At query time, this is how the flow should be 
    -> User types in a response in our UI
    -> /api/chat request gets fired off
    -> goes to the agent
    -> tool decides between the two engines (SQL and PDF)
    -> capture our evidence output
    -> return the AgentResponse back to the user

**Returning Data for both sql and pdf back to the user**
SQL - has exact rows (don't need to do anything, no score here)
PDF - because the number of documents is quite small, don't need to deal with scale. So we would do cosine similarity here. We can play with this but would stick with this i think. would just use k nearest neighbors for searching part.
- an additional step would be to add a reranker depending on time and cost?

**API Python library and UI Picking**
For Python API going to use Fast API and for UI i'm going to use Streamlit.

**Fireworks Model config and choices**
- Embedding and Reranking
So for the embedding model capabilities, fireworks has these models here: https://docs.fireworks.ai/guides/querying-embeddings-models#reranking-documents

https://app.fireworks.ai/models?type=embeddings&capability=serverless

From this link here, and from the model availability here, it seems qwen3-embedding-8b is the best choice for both reranking 
and embedding

**EVALS**
TODO (the hardest part)




