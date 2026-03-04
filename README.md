# Cyber Ireland 2022 Report Agent (Backend)

Headless Python backend that ingests the **Cyber Ireland 2022 PDF report**, builds a searchable knowledge index, and answers complex questions with:

- grounded retrieval
- step-by-step reasoning logs
- page-level citations
- deterministic mathematical calculations (for tasks like CAGR)

## Assignment Coverage

This implementation covers all required system components:

- ETL pipeline (PDF download, text/table extraction, cleaning, chunking)
- Embedding generation
- Vector database indexing and search
- Autonomous agent orchestration with tools
- Tooling for retrieval, citation extraction, and math
- FastAPI backend with `POST /query`
- Trace logging of reasoning steps
- Tests and demo query runner

---

## Architecture

### High-level flow

1. **Ingestion pipeline** processes the PDF offline.
2. **Chunks + metadata** are embedded and indexed in the vector store backend.
3. Runtime queries hit `POST /query`.
4. Agent plans actions and calls tools (retrieve/calculate/cite).
5. API returns answer with citations and reasoning trace.

### ASCII architecture diagram

```text
                           +-----------------------------+
                           |  Cyber Ireland 2022 PDF     |
                           +--------------+--------------+
                                          |
                                          v
                          +---------------+----------------+
                          | ETL Pipeline                    |
                          | - Text extraction               |
                          | - Table extraction              |
                          | - Cleaning                      |
                          | - Chunking + metadata tagging   |
                          +---------------+----------------+
                                          |
                         +----------------+----------------+
                         |                                 |
                         v                                 v
              +----------+----------+           +----------+----------+
              | Chunk/Metadata Store|           | Embedding Generator |
              | (JSONL)             |           | (MiniLM)            |
              +----------+----------+           +----------+----------+
                         |                                 |
                         +----------------+----------------+
                                          |
                                          v
                               +----------+----------+
                              | Vector Database      |
                              | (ChromaDB or local   |
                              |  JSON fallback)      |
                              +----------+----------+
                                          |
============================== RUNTIME QUERY PATH ==============================
                                          |
Client ---> POST /query ---> +------------+-------------+
                             | FastAPI + Query Agent    |
                             | (plan -> tools -> answer)|
                             +------+-----------+-------+
                                    |           |
                                    |           v
                                    |   +-------+--------+
                                    |   | Calculator Tool |
                                    |   +-------+--------+
                                    |
                                    v
                          +---------+----------+
                          | Retrieval Tool      |
                          +---------+----------+
                                    |
                                    v
                          +---------+----------+
                          | Citation Tool       |
                          +---------+----------+
                                    |
                                    v
                             JSON Response:
                      answer, citations, page, reasoning_steps
```

---

## Repository Structure

```text
cybersecurity-report-agent/
├── app/
│   ├── agents/          # Agent orchestration, LLM wrapper, tracing
│   ├── api/             # FastAPI routes and dependency wiring
│   ├── config/          # Settings and logging config
│   ├── etl/             # PDF loading, extraction, chunking pipeline
│   ├── models/          # Shared Pydantic schemas
│   ├── retrieval/       # Embeddings, vector DB, retriever, indexer
│   └── tools/           # Retrieval/citation/calculator tools
├── data/
│   ├── raw/             # Downloaded source PDF
│   ├── processed/       # Chunk JSONL
│   └── vectordb/        # Vector index files (generated locally)
├── scripts/
│   ├── ingest_report.py # ETL + indexing entrypoint
│   └── demo_queries.py  # Runs required test queries
├── tests/               # Unit/API/agent tests
├── .env.example
├── requirements.txt
└── README.md
```

---

## Technology Stack and Rationale

- **Python 3.11+**: mature ecosystem for LLM + data + APIs.
- **pdfplumber + PyMuPDF**: robust text extraction with fallback; table extraction via pdfplumber.
- **rapidocr-onnxruntime**: OCR fallback for chart-heavy/scanned page regions.
- **sentence-transformers/all-MiniLM-L6-v2**: fast, high-value embedding model for semantic retrieval.
- **Vector store backend**: ChromaDB when installed, otherwise a local JSON cosine-similarity fallback.
- **OpenAI API (optional)**: improves planning/synthesis; system still has deterministic fallback.
- **Custom tool-calling agent**: transparent and controllable multi-step orchestration.
- **FastAPI**: modern, typed backend framework with built-in OpenAPI docs.
- **Python logging**: step-level execution traces persisted to logs/app.log.

---

## Setup

### Prerequisites

- Python **3.12 recommended** (3.11+ supported)
- Run all commands from the repository root: `cybersecurity-report-agent/`

### 1) Create virtual environment

#### Windows (recommended)

```bash
py -3.12 -m venv .venv312
```

#### macOS / Linux

```bash
python3.12 -m venv .venv
```

### 2) Activate virtual environment

#### Windows Git Bash

```bash
source .venv312/Scripts/activate
```

#### Windows PowerShell

```powershell
.\.venv312\Scripts\Activate.ps1
```

#### macOS / Linux

```bash
source .venv/bin/activate
```

### 3) Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4) Configure environment file

#### Git Bash / macOS / Linux

```bash
cp .env.example .env
```

#### Windows PowerShell

```powershell
Copy-Item .env.example .env
```

`OPENAI_API_KEY` is optional. Without it, the system still runs end-to-end using deterministic planning/synthesis logic.

Optional `.env` values:

```env
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4o-mini
ENABLE_OCR_FALLBACK=true
OCR_DPI=180
```

---

## Quick Start (Fresh Clone)

After setup, run these in order:

```bash
python -m scripts.ingest_report
python -m pytest -q
python -m scripts.demo_queries
python -m uvicorn app.main:app --reload --port 8000
```

If you do not want to activate the environment, you can run with explicit interpreter path (Windows):

```bash
.venv312/Scripts/python -m scripts.ingest_report
.venv312/Scripts/python -m pytest -q
.venv312/Scripts/python -m scripts.demo_queries
.venv312/Scripts/python -m uvicorn app.main:app --reload --port 8000
```

---

## Run Ingestion (ETL + Index)

```bash
python -m scripts.ingest_report
```

By default, ingestion uses OCR fallback to recover additional text from chart-heavy pages.

If you want faster ingestion (or need to disable OCR temporarily):

Git Bash / macOS / Linux:

```bash
ENABLE_OCR_FALLBACK=false python -m scripts.ingest_report
```

Windows PowerShell:

```powershell
$env:ENABLE_OCR_FALLBACK="false"
python -m scripts.ingest_report
Remove-Item Env:ENABLE_OCR_FALLBACK
```

Optional force re-download:

```bash
python -m scripts.ingest_report --force-download
```

Artifacts generated:

- `data/raw/cyber_ireland_2022.pdf`
- `data/processed/chunks.jsonl`
- `data/vectordb/*`

---

## Run API Server

```bash
python -m uvicorn app.main:app --reload --port 8000
```

Endpoints:

- `GET /health`
- `POST /query`

Quick health check:

```bash
python -c "import httpx; r=httpx.get('http://127.0.0.1:8000/health', timeout=30); print(r.status_code); print(r.text)"
```

---

## API Contract

### Request

```json
{
  "query": "What is the total number of jobs reported, and where exactly is this stated?"
}
```

### Response

```json
{
  "answer": "...",
  "citations": [
    {
      "chunk_id": "p033-txt000",
      "page_number": 33,
      "quote": "The cybersecurity sector employs 7,351 people...",
      "source_type": "text"
    }
  ],
  "page": [33],
  "reasoning_steps": [
    {
      "step_number": 1,
      "action": "plan",
      "detail": "Agent created tool execution plan.",
      "tool_input": {"query": "..."},
      "tool_output": {"plan": [...]},
      "timestamp_utc": "2026-..."
    }
  ]
}
```

---

## Run Required Demo Queries

```bash
python -m scripts.demo_queries
```

Queries included:

1. Verification: total jobs + exact location
2. Data synthesis: South-West vs national pure-play concentration
3. Forecast calculation: CAGR from 2022 baseline to 2030 target

Expected behavior in deterministic mode (no OpenAI key):

- Query 1 should report approximately **7,351** jobs with page-level citations.
- Query 3 should report CAGR around **11.05%**.
- Query 2 uses OCR-augmented extraction and may improve evidence recall, but can still return an explicit "insufficient evidence" message if no reliable South-West vs national numeric values are extractable.

---

## Testing

```bash
python -m pytest -q
```

Included tests:

- calculator correctness
- chunk metadata integrity
- citation extraction behavior
- agent CAGR calculation flow
- API response contract

---

## Logging and Traceability

- Application logs are written to `logs/app.log`.
- Each query includes `reasoning_steps` in the API response.
- Tool invocations are recorded (retrieval hit IDs, calculator output, citation selection).

---

## Troubleshooting

### `No module named uvicorn`

You are likely using the wrong Python interpreter. Activate the correct venv and retry:

```bash
source .venv312/Scripts/activate
python -m uvicorn app.main:app --reload --port 8000
```

### `Activate.ps1` command fails in Git Bash

`Activate.ps1` is for PowerShell. In Git Bash, use:

```bash
source .venv312/Scripts/activate
```

### `deactivate: command not found`

You are not currently in an active virtual environment in that shell session.

### `chromadb not available; using local JSON vector index backend`

This is expected on environments where ChromaDB is not installed. The app automatically uses the built-in local JSON cosine-similarity backend.

### OCR is slow on my machine

OCR improves extraction quality but increases ingestion time. You can disable it:

Git Bash / macOS / Linux:

```bash
ENABLE_OCR_FALLBACK=false python -m scripts.ingest_report
```

Windows PowerShell:

```powershell
$env:ENABLE_OCR_FALLBACK="false"
python -m scripts.ingest_report
Remove-Item Env:ENABLE_OCR_FALLBACK
```

---

## Limitations

- Table extraction quality depends on source PDF formatting.
- OCR output from charts can still be noisy or incomplete for some pages.
- LLM quality and cost depend on configured model/provider.
- Current citation extraction is heuristic sentence selection from retrieved chunks.

---

## Future Improvements

- Add reranker model for improved retrieval precision.
- Add richer citation alignment (character offsets).
- Integrate OpenTelemetry/Langfuse for distributed tracing.
- Add evaluation harness with expected answer assertions for benchmark queries.
