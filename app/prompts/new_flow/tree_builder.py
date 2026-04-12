"""
Tree builder prompts — New Flow (PageIndex Vectorless RAG).

Used in: app/services/page_index/service.py → build_tree()
Purpose: Instruct LLM to parse a Markdown document and produce a hierarchical
         JSON tree of sections with semantic summaries and nodeIds.
LLM config: temperature=0.1, max_tokens=32768 (large legal docs need room; Gemini 2.5 Flash max is 65536)
"""

SYSTEM_PROMPT = """You are analyzing a legal/financial document. Your task is to build a complete, deeply nested hierarchical table of contents with semantic summaries for each section.

For each node in the tree provide:
- nodeId: unique dot-notation identifier (e.g. "1", "1.1", "1.1.2", "1.1.2.1")
- title: exact section heading as it appears in the document, OR a descriptive title you create for a logical sub-topic
- summary: 2-3 sentence description of WHAT this section covers — be specific about topics, definitions, rates, thresholds, penalties, and rules. Do NOT write vague summaries like "this section discusses provisions".
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
    },
    {
      "nodeId": "2",
      "title": "Tax Incentives",
      "summary": "Covers all tax concessions and exemptions for industries, investments, and special zones.",
      "page_range": [10, 18],
      "children": [
        {
          "nodeId": "2.1",
          "title": "Incentives in Income Tax Rates",
          "summary": "Details income tax rate concessions based on employment, geography, investment, and industry type.",
          "page_range": [10, 15],
          "children": [
            {
              "nodeId": "2.1.1",
              "title": "Employment-based tax rebates",
              "summary": "10-30% tax rebate for industries employing 100-1000+ Nepalese citizens, with additional 10% for 33%+ women employees.",
              "page_range": [10, 11],
              "children": []
            },
            {
              "nodeId": "2.1.2",
              "title": "Hydropower project exemptions",
              "summary": "100% income tax exemption for 10-15 years for hydropower projects, followed by 50% rebate for 5-6 years. Applies to generation, transmission, and distribution.",
              "page_range": [12, 12],
              "children": []
            },
            {
              "nodeId": "2.1.3",
              "title": "Geography-based incentives",
              "summary": "70-100% tax rate concessions for 10-15 years for industries in undeveloped and underdeveloped areas.",
              "page_range": [13, 14],
              "children": []
            }
          ]
        }
      ]
    }
  ]
}"""

SYSTEM_PROMPT_NE = """तपाईं एक कानुनी/वित्तीय नेपाली कागजात विश्लेषण गर्दै हुनुहुन्छ। प्रत्येक खण्डको लागि गहिरो पदानुक्रमिक संरचना र नेपालीमा सारांश प्रदान गर्नुहोस्।

प्रत्येक node मा निम्न जानकारी राख्नुहोस्:
- nodeId: अद्वितीय पहिचानकर्ता (जस्तै "१", "१.१", "१.१.२", "१.१.२.१")
- title: कागजातमा जस्तो छ त्यस्तै खण्डको शीर्षक, वा तपाईंले बनाउनुभएको वर्णनात्मक शीर्षक
- summary: यस खण्डमा के छ भनेर २-३ वाक्यमा विस्तृत विवरण — परिभाषाहरू, दरहरू, सीमाहरू, दण्डहरू र नियमहरू उल्लेख गर्नुहोस्
- page_range: [सुरु_पृष्ठ, अन्त्य_पृष्ठ]
- children: बाल nodes को सूची

महत्वपूर्ण — विस्तृतता नियमहरू:
- प्रत्येक leaf node अधिकतम २ पृष्ठ मात्र हुनुपर्छ। ३+ पृष्ठ भएमा children बनाउनुहोस्।
- एउटा खण्डमा विभिन्न विषयहरू छन् भने (जस्तै विभिन्न उद्योगका कर दरहरू), प्रत्येक विषयको लागि छुट्टै child node बनाउनुहोस्।
- तालिका वा सूचीमा विभिन्न वर्गहरू छन् भने, प्रत्येक वर्ग/समूहको लागि एक child node बनाउनुहोस्।
- गहिरो रूख (४-५ तह) बनाउनुहोस्। थप विशिष्ट nodes = राम्रो खोजी।

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
