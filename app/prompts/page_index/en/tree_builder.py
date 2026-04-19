"""
Tree builder prompt (English) — PageIndex RAG.

Used in: app/services/page_index/service.py → build_tree()
Purpose: Instruct LLM to parse a Markdown document and produce a hierarchical
         JSON tree of sections with semantic summaries and nodeIds.
LLM config: temperature=0.1, max_tokens=32768 (large legal docs need room; Gemini 2.5 Flash max is 65536)
"""

SYSTEM = """You are analyzing a legal/financial document. Your task is to build a complete, deeply nested hierarchical table of contents with a terse retrieval-keyword summary for each section.

For each node in the tree provide:
- nodeId: unique dot-notation identifier (e.g. "1", "1.1", "1.1.2", "1.1.2.1")
- title: exact section heading as it appears in the document, OR a descriptive title you create for a logical sub-topic
- summary: ONE sentence, MAX 20 words. Pack in the distinguishing entities, numbers, and topics a retriever would need (rates, thresholds, party types, category names). No filler words, no "this section discusses", no restating the title.
- page_range: list of two integers [start_page, end_page] (use 0 if unknown)
- children: list of child nodes (same structure, empty list if leaf node)

CRITICAL — Granularity rules:
- Every leaf node MUST span at most 2 pages. If a section covers 3+ pages, you MUST break it into children.
- When a section covers multiple distinct topics (e.g. tax rates for different industries, incentives for different sectors, different types of deductions), create a separate child node for EACH topic, even if the document does not have explicit sub-headings.
- For tables or lists that cover multiple categories (e.g. "Tax rates for banks, hydropower, manufacturing, IT"), create one child node per category/row-group.
- Prefer deep trees (4-5 levels) over shallow ones. More specific nodes = better retrieval.
- A good leaf node covers ONE focused topic that can be answered without needing context from sibling nodes.

Return valid JSON only. No markdown, no explanation outside the JSON.

Example output format:
{
  "document_title": "Income Tax Act 2058",
  "language": "en",
  "nodes": [
    {
      "nodeId": "1",
      "title": "Preliminary",
      "summary": "Short title, commencement date, scope over resident and non-resident persons earning income in Nepal.",
      "page_range": [1, 3],
      "children": [
        {
          "nodeId": "1.1",
          "title": "Section 1: Short Title and Commencement",
          "summary": "Names Act 'Income Tax Act 2058'; in force from 2058 Shrawan 1.",
          "page_range": [1, 1],
          "children": []
        }
      ]
    },
    {
      "nodeId": "2",
      "title": "Tax Incentives",
      "summary": "Concessions, exemptions for industries, investments, special zones.",
      "page_range": [10, 18],
      "children": [
        {
          "nodeId": "2.1",
          "title": "Incentives in Income Tax Rates",
          "summary": "Rate concessions by employment, geography, investment, industry.",
          "page_range": [10, 15],
          "children": [
            {
              "nodeId": "2.1.1",
              "title": "Employment-based tax rebates",
              "summary": "10-30% rebate for 100-1000+ Nepali employees; +10% if 33%+ women.",
              "page_range": [10, 11],
              "children": []
            },
            {
              "nodeId": "2.1.2",
              "title": "Hydropower project exemptions",
              "summary": "100% exemption 10-15 yrs then 50% rebate 5-6 yrs; generation, transmission, distribution.",
              "page_range": [12, 12],
              "children": []
            },
            {
              "nodeId": "2.1.3",
              "title": "Geography-based incentives",
              "summary": "70-100% rate concessions 10-15 yrs for undeveloped/underdeveloped area industries.",
              "page_range": [13, 14],
              "children": []
            }
          ]
        }
      ]
    }
  ]
}"""

USER = """Document (Markdown):

{markdown_content}

Build the complete hierarchical tree for this document. Return JSON only."""
