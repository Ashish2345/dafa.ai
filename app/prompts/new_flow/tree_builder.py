"""
Tree builder prompts — New Flow (PageIndex Vectorless RAG).

Used in: app/services/page_index/service.py → build_tree()
Purpose: Instruct LLM to parse a Markdown document and produce a hierarchical
         JSON tree of sections with semantic summaries and nodeIds.
LLM config: temperature=0.1, max_tokens=8192 (large documents need room)
"""

SYSTEM_PROMPT = """You are analyzing a legal/financial document. Your task is to build a complete hierarchical table of contents with semantic summaries for each section.

For each node in the tree provide:
- nodeId: unique dot-notation identifier (e.g. "1", "1.1", "1.1.2")
- title: exact section heading as it appears in the document
- summary: 2-3 sentence description of WHAT this section covers — be specific about topics, definitions, rates, thresholds, penalties, and rules. Do NOT write vague summaries like "this section discusses provisions".
- page_range: list of two integers [start_page, end_page] (use 0 if unknown)
- children: list of child nodes (same structure, empty list if leaf node)

Return valid JSON only. No markdown, no explanation outside the JSON.

Example output format:
{
  "document_title": "Income Tax Act 2058",
  "language": "en",
  "nodes": [
    {
      "nodeId": "1",
      "title": "Preliminary",
      "summary": "Defines the short title and commencement date of the Act. Establishes that the Act applies to all resident and non-resident persons earning income in Nepal.",
      "page_range": [1, 3],
      "children": [
        {
          "nodeId": "1.1",
          "title": "Section 1: Short Title and Commencement",
          "summary": "Names the Act as 'Income Tax Act 2058' and specifies it came into force on 2058 Shrawan 1.",
          "page_range": [1, 1],
          "children": []
        }
      ]
    }
  ]
}"""

SYSTEM_PROMPT_NE = """तपाईं एक कानुनी/वित्तीय नेपाली कागजात विश्लेषण गर्दै हुनुहुन्छ। प्रत्येक खण्डको लागि पदानुक्रमिक संरचना र नेपालीमा सारांश प्रदान गर्नुहोस्।

प्रत्येक node मा निम्न जानकारी राख्नुहोस्:
- nodeId: अद्वितीय पहिचानकर्ता (जस्तै "१", "१.१", "१.१.२")
- title: कागजातमा जस्तो छ त्यस्तै खण्डको शीर्षक
- summary: यस खण्डमा के छ भनेर २-३ वाक्यमा विस्तृत विवरण — परिभाषाहरू, दरहरू, सीमाहरू, दण्डहरू र नियमहरू उल्लेख गर्नुहोस्
- page_range: [सुरु_पृष्ठ, अन्त्य_पृष्ठ]
- children: बाल nodes को सूची

केवल valid JSON मात्र फर्काउनुहोस्। JSON बाहिर कुनै explanation नराख्नुहोस्।"""

USER_PROMPT = """Document (Markdown):

{markdown_content}

Build the complete hierarchical tree for this document. Return JSON only."""

USER_PROMPT_NE = """कागजात (Markdown):

{markdown_content}

यस कागजातको पूर्ण पदानुक्रमिक संरचना बनाउनुहोस्। केवल JSON मात्र फर्काउनुहोस्।"""


def get_prompts(language: str = "en") -> tuple[str, str]:
    """Return (system_prompt, user_prompt_template) for the given language."""
    if language == "ne":
        return SYSTEM_PROMPT_NE, USER_PROMPT_NE
    return SYSTEM_PROMPT, USER_PROMPT
