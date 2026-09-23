# Dafa.ai

A FastAPI backend that answers questions about Nepali law and regulation. It collects legal and regulatory PDFs, parses them (running OCR on scanned pages), turns each document into a tree of sections, and answers questions with citations back to the exact page.

## What it does

- **Ingestion.** Crawlers driven by a source registry fetch documents from regulator and gazette sites. Each source uses one parse profile (`index_pdf_links`, `giwms_volume_list`, `local_directory`), so adding a new source only needs a config change. A change detector finds new or changed publications, and a health check flags sources that have gone stale.
- **Parsing.** Text is taken from the PDF text layer where one exists. Scanned pages go to cloud OCR (Google Vision or AWS Textract). Documents are then parsed into act → chapter → section trees in English and Nepali.
- **Retrieval.** Page-index retrieval walks the document tree. Hybrid search combines BM25 with vector search in Qdrant, and results are re-ranked with a cross-encoder.
- **Answering.** Gemini writes the answer and cites the section and page it relied on. A "studio" layer classifies each query and sends it to the right pipeline.
- **Accounts.** JWT auth, TOTP two-factor auth with backup codes, active-session management, invites, and usage limits per plan (`config/plans.yaml`).

## Stack

Python 3.11+, FastAPI, MongoDB, Qdrant, Google Gemini, OpenAI embeddings, sentence-transformers, PyMuPDF, uv, Docker.

## Layout

```
app/
  api/v1/endpoints/   auth, documents, query, studio, user, changes, health
  services/
    injection/        source crawlers, change detection, source health
    ingestion/        parse → tree → index pipeline
    retrieval/        page-index + hybrid retrieval
    llm/              Gemini client
    studio/           query classification, routing, answer aggregation
  db/                 Mongo repositories, schema validators, seeds
config/               sources.yaml (source registry seed), plans.yaml
scripts/              inject.py (run crawlers), parse_tree.py, backfill
```

## Running locally

```bash
cp .env.example .env          # fill in MongoDB and provider keys
uv sync
uv run uvicorn app.main:app --reload --port 8000
```

Or with Docker:

```bash
docker build -t dafa-ai .
docker run --env-file .env -p 8000:8000 dafa-ai
```

API docs are served at `http://localhost:8000/docs`. A Postman collection is in `dafa-ai.postman_collection.json`.

## Credentials

Credentials never go in the repo. Provide them through `.env`. For Google Cloud, set `GOOGLE_APPLICATION_CREDENTIALS` to a key file stored outside the repository.
