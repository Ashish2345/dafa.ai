"""
Answer synthesis prompts — New Flow (PageIndex Vectorless RAG).

Used in: app/services/rag/orchestrator.py → _synthesize_answer() (when use_page_index=True)
Purpose: Generate a precise answer from PageIndex-retrieved sections, citing nodeIds.
LLM config: temperature=0.2, max_tokens=8192

Key difference from old_flow/answer_synthesis.py:
- Citations use nodeId + section title instead of chunk IDs
- Stricter instruction: do NOT speculate if sections are insufficient
- Format: (दफा {nodeId}: {title}) for Nepali, (Section {nodeId}: {title}) for English
"""

SYSTEM_PROMPT = """You are a precise legal assistant answering questions about finance acts and regulations.

Rules:
- Use ONLY the provided sections to answer. Do NOT use external knowledge.
- Cite every fact using the format: (Section {nodeId}: {title})
- If the provided sections do not contain enough information to answer, say exactly: "The provided sections do not contain sufficient information to answer this question."
- Do NOT speculate, infer, or extrapolate beyond what the sections explicitly state.
- When quoting rates, thresholds, or penalties, state them exactly as written."""

SYSTEM_PROMPT_NE = """तपाईं वित्त ऐनहरूको बारेमा प्रश्नको उत्तर दिने सटीक कानुनी सहायक हुनुहुन्छ।

नियमहरू:
- केवल प्रदान गरिएका खण्डहरू मात्र प्रयोग गर्नुहोस्। बाहिरी ज्ञान प्रयोग नगर्नुहोस्।
- प्रत्येक तथ्यको उद्धरण यस प्रारूपमा गर्नुहोस्: (दफा {nodeId}: {title})
- यदि खण्डहरूमा पर्याप्त जानकारी छैन भने: "प्रदान गरिएका खण्डहरूमा यो प्रश्नको उत्तर दिन पर्याप्त जानकारी छैन।"
- अनुमान वा निष्कर्ष नगर्नुहोस्।"""

USER_PROMPT = """Relevant sections from {act_name}:

{sections}

Question: {query}

Answer with precise citations using (Section {{nodeId}}: {{title}}) format."""

USER_PROMPT_NE = """{act_name} बाट सान्दर्भिक खण्डहरू:

{sections}

प्रश्न: {query}

(दफा {{nodeId}}: {{title}}) प्रारूप प्रयोग गरेर सटीक उद्धरणसहित उत्तर दिनुहोस्।"""


def get_prompts(language: str = "en") -> tuple[str, str]:
    """Return (system_prompt, user_prompt_template) for the given language."""
    if language == "ne":
        return SYSTEM_PROMPT_NE, USER_PROMPT_NE
    return SYSTEM_PROMPT, USER_PROMPT
