"""
New flow prompts — PageIndex Vectorless RAG pipeline.

Flow: PDF → OCR → Markdown → PageIndex Tree Build (LLM summaries)
      → MongoDB Tree Storage → LLM Tree Navigation → Gemini Answer with nodeId citations
"""
