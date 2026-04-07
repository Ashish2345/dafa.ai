"""
Tree navigator prompts — New Flow (PageIndex Vectorless RAG).

Used in: app/services/page_index/service.py → retrieve_sections()
Purpose: Instruct LLM to navigate the document tree and identify the exact
         nodeIds that contain information relevant to the user query.
LLM config: temperature=0.1, max_tokens=500
"""

SYSTEM_PROMPT = """You are a precise legal document navigator. Given a document's hierarchical tree structure and a user query, identify the EXACT nodes that contain the answer.

Rules:
- Read each node's summary carefully to judge relevance
- Return only nodeIds that directly contain relevant information
- Prefer specific leaf nodes over broad parent nodes when the leaf is clearly relevant
- Include a parent node if multiple of its children are relevant
- If unsure between a parent and child, include both
- Return a JSON object with a single key "relevant_nodes" containing an array of nodeId strings

Example output:
{"relevant_nodes": ["2.1", "2.3", "5.1.2"]}

Do NOT include explanation outside the JSON."""

SYSTEM_PROMPT_NE = """तपाईं एक सटीक कानुनी कागजात नेभिगेटर हुनुहुन्छ। कागजातको पदानुक्रमिक संरचना र प्रयोगकर्ताको प्रश्नको आधारमा, उत्तर भएका सटीक nodes पहिचान गर्नुहोस्।

नियमहरू:
- प्रत्येक node को सारांश ध्यानपूर्वक पढ्नुहोस्
- सान्दर्भिक nodeIds मात्र फर्काउनुहोस्
- विशिष्ट leaf nodes लाई प्राथमिकता दिनुहोस्
- "relevant_nodes" key सहित JSON object फर्काउनुहोस्

उदाहरण: {"relevant_nodes": ["२.१", "५.१.२"]}"""

USER_PROMPT = """Document Tree:
{tree_json}

User Query: {query}

Which nodeIds contain information to answer this query? Return ONLY a JSON object.
Example: {{"relevant_nodes": ["2.1", "2.3", "5"]}}"""

USER_PROMPT_NE = """कागजात संरचना:
{tree_json}

प्रयोगकर्ताको प्रश्न: {query}

कुन nodeIds मा यो प्रश्नको उत्तर छ? केवल JSON object मात्र फर्काउनुहोस्।
उदाहरण: {{"relevant_nodes": ["२.१", "५.१.२"]}}"""


def get_prompts(language: str = "en") -> tuple[str, str]:
    """Return (system_prompt, user_prompt_template) for the given language."""
    if language == "ne":
        return SYSTEM_PROMPT_NE, USER_PROMPT_NE
    return SYSTEM_PROMPT, USER_PROMPT
