INGESTION FLOW — POST /documents/upload

POST /upload
│
├─ 1. FileHandler saves PDF to temp disk
├─ 2. DocumentRepository.save_initial()  → MongoDB documents{}
│       status: "processing", progress_step: "Queued"
│
└─ BackgroundTask: _run_ingestion_background()
        │
        ├─ STEP 1  "Parsing document (OCR)..."
        │   PDFParser._pre_process_sync()
        │     ├─ PDFReader  → reads all pages with FitzBackend
        │     ├─ DigitalOCR (if text-based) / GoogleVisionOCR (if scanned)
        │     └─ returns raw_ocr: List[DataFrame], page_scalars
        │
        ├─ STEP 2  "Converting pages to Markdown..."
        │   IngestionService.gather_document_data()
        │   DocumentProcessor.process_to_markdown()
        │     └─ OCRInputParser.parse()
        │           626 pages → layout_conserved_with_lineno
        │           → one big markdown string (e.g. 826,276 chars)
        │
        ├─ STEP 3  "Extracting metadata..."
        │   MetadataExtractor.extract_metadata()
        │     → title, author, language
        │
        ├─ STEP 4  "Saving PDF to storage..."
        │   FileStorageRepository.save_pdf() → GridFS
        │
        ├─ STEP 5  PageIndexStrategy.ingest()   ← THE BIG ONE
        │   │
        │   ├─ TreeBuilder.split_chunks(markdown)
        │   │     826K chars → [chunk1 350K, chunk2 350K, chunk3 126K]
        │   │
        │   ├─ FOR EACH CHUNK  (progress: "Building index tree: chunk i/3...")
        │   │   TreeBuilder.build_chunk(chunk, language)
        │   │     │
        │   │     ├─ LLMService.call(prompt, add_warning=False, max_tokens=32768)
        │   │     │    Gemini 2.5 Flash
        │   │     │    System: "Build a complete hierarchical table of contents..."
        │   │     │    User:   chunk of markdown
        │   │     │
        │   │     └─ Returns subtree JSON:
        │   │          {
        │   │            "document_title": "Income Tax Act 2058",
        │   │            "nodes": [
        │   │              { "nodeId":"1", "title":"Preamble", "summary":"...",
        │   │                "page_range":[1,2], "children":[...] }
        │   │            ]
        │   │          }
        │   │
        │   ├─ _merge_trees([subtree1, subtree2, subtree3])
        │   │     → combined nodes list, title from first chunk
        │   │
        │   ├─ _attach_char_offsets(nodes, full_markdown)
        │   │     For each node: regex `#{1,6}[^\n]*{title}` in markdown
        │   │     → node.start_char, node.end_char  (slice positions)
        │   │
        │   └─ PageIndexRepository.save_tree()
        │         page_index_trees{}   → { document_id, tree: {nodes}, language }
        │         page_index_content{} → { document_id, markdown: "full 826K chars" }
        │
        └─ STEP 6  "Saving document record..."
            DocumentRepository.save()
              → documents{} status: "completed", progress_step: "Done"
RAG QUERY FLOW — POST /query

POST /query  { "query": "What is income tax for 10k/month?", "strategy": "page_index" }
│
└─ PageIndexStrategy.retrieve(query, top_k=5)
        │
        ├─ PageIndexRepository.list_document_ids()
        │     → ["doc-uuid-1", "doc-uuid-2", ...]
        │
        └─ FOR EACH document:
              │
              ├─ repo.get_tree(doc_id)     → tree JSON  (from page_index_trees)
              ├─ repo.get_markdown(doc_id) → full markdown (from page_index_content)
              │
              └─ SectionRetriever.retrieve(query, tree, markdown)
                    │
                    ├─ _build_compact_tree(tree)
                    │     strips start/end chars → { nodeId, title, summary, children }
                    │     (lightweight — only what the LLM needs to navigate)
                    │
                    ├─ LLM CALL 1 — Tree Navigation (max_tokens=500)
                    │   LLMService.call(tree_json + query)
                    │   System: "You are a precise legal document navigator..."
                    │   User:   compact tree JSON + query
                    │   Returns: { "relevant_nodes": ["2.1", "3.4", "5"] }
                    │
                    ├─ _build_node_lookup(tree)  → flat map { nodeId → node }
                    │
                    └─ FOR EACH relevant nodeId:
                          _extract_node_text(node, markdown)
                            ├─ Primary:  markdown[node.start_char : node.end_char]
                            └─ Fallback: regex #{1,6}[^\n]*{title}...next heading
                                         → node.summary (last resort)

        all_chunks sorted by score (1.0, 0.9, 0.8...)
        → top_k returned as RetrievedChunk[]

        │
        └─ LLM CALL 2 — Answer Synthesis  (max_tokens=8192)
              LLMService.synthesize(query, chunks)
              System: "Use ONLY the provided sections. Cite (Section nodeId: title)..."
              User:   [Section 1] Act: Income Tax Act\n Node: 2.1\n Content: <raw text>
              Returns: "Based on Section 2.1: Remuneration Payments, the tax rate for..."

Final response:
{
  "query": "...",
  "answer": "LLM synthesized answer with citations",
  "chunks": [ { "text": "<raw markdown slice>", "source": { "node_id": "2.1" }, "score": 1.0 } ],
  "sources": [ { "document_name": "Income Tax Act 2058", "pages": [2,2] } ]
}