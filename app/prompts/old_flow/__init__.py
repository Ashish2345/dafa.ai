"""
Old flow prompts — Traditional Vector RAG pipeline.

Flow: PDF → OCR → Markdown → Chunk → OpenAI Embeddings → Qdrant
      → BM25 Hybrid Search → Cross-encoder Rerank → Gemini Answer
"""
