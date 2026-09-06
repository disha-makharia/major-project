# DataMind — Multi-Agent RAG for Data Analytics

DataMind is an AI-powered data analytics platform. Upload CSV, JSON, or
Parquet datasets and company documents (PDF/TXT/DOCX), then ask questions in
plain English. A team of specialist agents, coordinated by a LangGraph
orchestrator, figures out what data is relevant, writes and runs SQL against
it, searches your documents for supporting context, analyzes the results,
builds a chart when it's useful, validates the answer, and returns a
business-friendly response — no SQL knowledge required.

## Architecture

```
User
  |
FastAPI
  |
LangGraph Orchestrator
  |
Supervisor Agent
  ├── Data Agent            (finds the relevant dataset/columns)
  ├── SQL Agent             (writes + runs DuckDB SQL, self-corrects on error)
  ├── RAG Agent             (semantic search over your documents)
  ├── Analysis Agent        (turns results into grounded insights)
  ├── Visualization Agent   (builds a chart only when one helps)
  └── Response Agent        (validates and writes the final answer)
  |
Ollama LLM
  |
DuckDB + Local Data Hub  +  ChromaDB RAG Pipeline
  |
Final validated response + visualization
```

The Supervisor decides, per question, whether the SQL and/or RAG branches
are even needed — it does not blindly run every agent for every question.

## Features

- Upload multiple CSV / JSON / Parquet datasets; each becomes a real DuckDB
  table with schema, row counts, and sample records available to every agent.
- Natural-language to SQL, executed on DuckDB, with a safety layer that
  blocks any non-read-only statement (`DROP`, `DELETE`, `UPDATE`, `INSERT`,
  `ALTER`, `CREATE`, `TRUNCATE`, ...) and an automatic retry/correction loop
  when generated SQL fails.
- Document RAG pipeline: PDF/TXT/DOCX → chunking → local embeddings →
  ChromaDB → semantic search, with source attribution on every retrieved
  chunk.
- Automatic chart generation (bar/line/pie/scatter/histogram) — only when a
  chart actually helps, straight from the real query result rows.
- Multi-layer validation before the answer is returned: did SQL succeed, are
  retrieved documents relevant, does every number in the final answer trace
  back to real data, does the chart match the underlying result.
- Fully configurable via `.env`: LLM model, embedding model, storage paths.
- Structured JSON logging across every agent, SQL statement, and graph
  transition (no secrets ever logged).

## Technology Stack

Python · FastAPI · LangGraph · Ollama · DuckDB · ChromaDB ·
sentence-transformers (local embeddings) · Pandas · PyArrow · Matplotlib ·
pypdf / python-docx · HTML/CSS/JavaScript frontend.

## Project Structure

```
datamind/
├── backend/
│   ├── main.py                 FastAPI app, static frontend mount
│   ├── config.py                Environment-driven settings
│   ├── requirements.txt
│   ├── api/routes.py             All REST endpoints
│   ├── agents/                   Supervisor, Data, SQL, RAG, Analysis,
│   │                              Visualization, Response agents + state
│   ├── graph/workflow.py         LangGraph StateGraph wiring + routing
│   ├── data/dataset_manager.py   CSV/JSON/Parquet ingestion + catalog
│   ├── rag/                      Document loader, chunker, embeddings,
│   │                              ChromaDB vector store, document manager
│   ├── database/duckdb_engine.py SQL execution engine
│   ├── visualization/            Chart type decision + rendering
│   ├── models/schemas.py         Pydantic request/response models
│   └── utils/                    Logging, SQL safety, Ollama client
├── frontend/                    index.html / style.css / app.js
├── data/
│   ├── datasets/                 sales.csv, products.csv (sample data)
│   ├── documents/                 sales_report.pdf (sample document)
│   └── chroma/                    ChromaDB persistent store (generated)
├── generated/charts/            Generated chart PNGs
├── tests/                       Pytest suite
├── .env.example
├── docker-compose.yml
└── README.md
```

## Installation

### 1. Python environment

Requires Python 3.11+.

```bash
cd datamind
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r backend/requirements.txt
```

### 2. Install and start Ollama

DataMind uses [Ollama](https://ollama.com) as its LLM provider — no cloud API
keys required.

```bash
# Install (see https://ollama.com/download for your OS)
curl -fsSL https://ollama.com/install.sh | sh

# Start the server (if not already running as a service)
ollama serve

# Pull a model (in another terminal) — pick one that fits your machine
ollama pull llama3.1
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env if you want a different OLLAMA_MODEL, EMBEDDING_MODEL, or paths.
```

`EMBEDDING_MODEL` defaults to `all-MiniLM-L6-v2` (via `sentence-transformers`)
and downloads automatically from Hugging Face the first time it's used —
this requires network access once; after that it's cached locally and needs
no network or paid API.

### 4. Start the backend

Run from the project root (`datamind/`) so the `backend` package resolves:

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

On startup, DataMind automatically ingests any files already sitting in
`data/datasets/` and `data/documents/` — including the bundled sample
`sales.csv`, `products.csv`, and `sales_report.pdf` — so you can start asking
questions immediately.

### 5. Open the frontend

Visit **http://localhost:8000/** — the backend serves the frontend directly,
no separate frontend server needed.

### Docker Compose (alternative)

```bash
docker compose up --build
# then, once ollama is up:
docker exec -it datamind-ollama ollama pull llama3.1
```

This starts Ollama and the backend together; the frontend is served from the
same backend container at http://localhost:8000/.

## Uploading Data

**Datasets** (sidebar → "Upload Dataset", or `POST /upload/dataset`): CSV,
JSON, or Parquet. Each upload is saved under `data/datasets/`, parsed with
Pandas/PyArrow, registered as a DuckDB table (sanitized from the filename),
and its schema (columns, types, row count, sample rows) is immediately
available to every agent.

**Documents** (sidebar → "Upload Document", or `POST /upload/document`):
PDF, TXT, or DOCX. Each document is text-extracted, chunked, embedded with
the local embedding model, and stored in ChromaDB for semantic search.

## Example Questions

With the bundled sample data (`sales.csv` / `products.csv` /
`sales_report.pdf`):

- "What were total sales?"
- "Which product generated the highest revenue?"
- "What were the monthly sales trends?"
- "Compare sales between regions."
- "What does the sales report say about the decline in Q3?"
- "Which region performed best?"
- "Why did sales decrease in Q3 and what does our sales report say about it?"

## API Endpoints

| Method | Path                  | Description                                   |
|--------|-----------------------|------------------------------------------------|
| POST   | `/upload/dataset`     | Upload a CSV/JSON/Parquet dataset               |
| POST   | `/upload/document`    | Upload a PDF/TXT/DOCX document into the RAG KB  |
| POST   | `/query`              | Ask a natural-language question                 |
| GET    | `/datasets`           | List uploaded datasets and their schemas        |
| GET    | `/documents`          | List ingested documents                         |
| GET    | `/health`             | Ollama/embedding/dataset/document status         |
| GET    | `/charts/{filename}`  | Fetch a generated chart image                   |

Interactive API docs: http://localhost:8000/docs

### `POST /query` request/response

```json
// Request
{ "question": "Which product generated the highest revenue?" }

// Response (abridged)
{
  "answer": "...",
  "plan": ["data_agent", "sql_agent", "analysis_agent", "visualization_agent", "response_agent"],
  "sql": { "query": "SELECT ...", "columns": [...], "rows": [...], "row_count": 5 },
  "retrieved_documents": [{ "source": "sales_report.pdf", "text": "...", "similarity": 0.42 }],
  "insights": ["'Laptop Pro 14' has the highest revenue at 1,060,800.00."],
  "visualization": { "generated": true, "chart_type": "bar", "chart_url": "/charts/chart_xxx.png" },
  "validation": { "passed": true, "checks": {...}, "issues": [] }
}
```

## Testing

```bash
cd datamind
source venv/bin/activate
pytest tests/ -v
```

The suite covers dataset upload/schema detection, SQL safety validation,
DuckDB execution, the RAG chunker/loader, agent state shape, the LangGraph
workflow's conditional routing, and the API endpoints end-to-end (with
Ollama calls skipped/mocked where the environment has no LLM running, so the
suite is runnable offline; anything requiring a live model is marked and
skips cleanly instead of failing).

## Troubleshooting

- **"Cannot reach Ollama"** — make sure `ollama serve` is running and
  `OLLAMA_BASE_URL` in `.env` points at it (default `http://localhost:11434`).
  `GET /health` reports the exact problem.
- **"Model not found"** — run `ollama pull <model>` for whatever
  `OLLAMA_MODEL` is set to.
- **Embedding model fails to load** — the first run needs network access to
  download the model from Hugging Face once; after that it's cached in
  `~/.cache/huggingface`. Set `EMBEDDING_MODEL` to a different local model if
  needed.
- **"Unsupported file type"** — only `.csv/.json/.parquet` for datasets and
  `.pdf/.txt/.docx` for documents are accepted.
- **SQL keeps failing** — the SQL Agent retries automatically (see
  `MAX_SQL_RETRIES`); if it still fails, the response will say so explicitly
  rather than fabricating a result.
- **No chart generated** — this is by design when the result shape (e.g. a
  single aggregate number, too many categories) wouldn't produce a
  meaningful chart.
- **Port already in use** — change `--port` on the `uvicorn` command and/or
  `APP_PORT` in `.env`.
