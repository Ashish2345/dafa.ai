"""
LLM re-ranking prompts — Old Flow (Vector RAG).

Used in: app/services/retrieval/reranker.py → _llm_rerank()
Purpose: Score each retrieved chunk's relevance to the query (0.0–1.0).
LLM config: temperature=0.1, max_tokens=200
Note: Disabled by default (use_llm_rerank=False) due to extra API cost.
      Only activated for high-stakes query intents: calculation, legal_interpretation.
"""

SYSTEM_PROMPT = """You are a relevance scorer for finance act documents.
Given a user query and a list of document chunks, score each chunk's relevance to the query.
Return ONLY a JSON array of scores (numbers between 0.0 and 1.0), one score per chunk, in order.
Example: [0.9, 0.7, 0.5, 0.3, 0.1]"""

SYSTEM_PROMPT_NE = """तपाईं वित्त ऐन कागजातहरूको लागि सान्दर्भिकता स्कोरर हुनुहुन्छ।
प्रयोगकर्ताको प्रश्न र कागजातका खण्डहरूको सूचीको आधारमा, प्रत्येक खण्डको सान्दर्भिकता स्कोर गर्नुहोस्।
केवल स्कोरहरूको JSON array मात्र फर्काउनुहोस् (0.0 देखि 1.0 सम्म), क्रमशः।
उदाहरण: [0.9, 0.7, 0.5, 0.3, 0.1]"""

USER_PROMPT = """User Query: {query}

Document Chunks:
{chunks_text}

Score each chunk's relevance to the query. Return ONLY a JSON array of scores."""

USER_PROMPT_NE = """प्रयोगकर्ताको प्रश्न: {query}

कागजातका खण्डहरू:
{chunks_text}

प्रत्येक खण्डको सान्दर्भिकता स्कोर गर्नुहोस्। केवल JSON array मात्र फर्काउनुहोस्।"""


def get_prompts(language: str = "en") -> tuple[str, str]:
    """Return (system_prompt, user_prompt_template) for the given language."""
    if language == "ne":
        return SYSTEM_PROMPT_NE, USER_PROMPT_NE
    return SYSTEM_PROMPT, USER_PROMPT
