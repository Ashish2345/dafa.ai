# Dafa.ai

A FastAPI backend that answers questions about Nepali law and regulation. You upload an act or regulation as a PDF. The service reads it (running OCR on scanned pages), has an LLM build a section-level index tree, and answers questions using only the sections it finds, citing the section and page.

## How it works

**Ingestion** (`POST /documents/upload`, runs as a background task)

1. **Read the PDF.** Pages with a text layer are read directly with PyMuPDF. Scanned pages go to OCR, with Google Vision as the default and AWS Textract and Azure also available.
2. **Convert to Markdown.** OCR output becomes Markdown with the original layout preserved, and the title, author and language are extracted.
3. **Store the original.** The PDF is saved to MongoDB GridFS.
4. **Build the index tree.** Gemini splits the Markdown into a hierarchy of act → chapter → section nodes, each with a title, a summary and character offsets. Documents in English and Nepali are both supported.

**Query** (`POST /query`, with a streaming variant)

1. **Find sections.** In the default `page_index` mode, the LLM reads a compact version of each document's tree and picks the relevant section nodes. The exact text of those nodes is then cut out of the stored Markdown.
2. **Answer.** Gemini writes the answer from those sections only and cites each one (`Section 2.1: …`). The response includes the source document, the page range and highlight data for the frontend.

A `vector` mode is also available: BM25 and embedding search in Qdrant, followed by a cross-encoder re-ranker. A "studio" layer classifies incoming questions and routes each one to the right pipeline.

**Around it:** JWT auth, chat history, follow-up suggestions, starred answers, feedback, and per-plan usage quotas. Crawlers (`scripts/inject.py`) pull source PDFs from Nepali regulators: IRD, NRB, the Law Commission, the Gazette, and the Finance Acts.

## Stack

Python 3.11+, FastAPI, MongoDB (+ GridFS), Google Gemini, Google Vision / AWS Textract OCR, Qdrant, sentence-transformers, PyMuPDF, uv. Deployed to a VM by GitHub Actions and run as a systemd service.

## Layout

```
app/
  api/v1/endpoints/   auth, documents, query, studio, user, parse, health
  core/pdf/           PDF reader + OCR backends (digital, Google, AWS, Azure)
  services/
    ingestion/        upload → OCR → markdown → index tree
    retrieval/        page_index (LLM tree navigation) and vector (hybrid) strategies
    query/            orchestration, synthesis, follow-ups, quotas
    studio/           query classification and routing
    injection/        source crawlers
  db/repositories/    MongoDB data access
flow.md               step-by-step ingestion and query flow
```

## Running locally

```bash
cp .env.example .env     # MongoDB connection + provider keys
uv sync
uv run uvicorn app.main:app --reload --port 5000
```

Interactive API docs are served at `http://localhost:5000/docs`.

Keep credentials out of the repo. Put them in `.env`, and for Google Cloud set `GOOGLE_APPLICATION_CREDENTIALS` to a file stored outside the repository.
