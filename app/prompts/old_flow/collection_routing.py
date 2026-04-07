"""
Collection routing prompts — Old Flow (Vector RAG).

Used in: app/services/rag/collection_router.py → route_query()
Purpose: Identify which Qdrant collection(s) a user query targets.
LLM config: temperature=0.1, max_tokens=200
"""

SYSTEM_PROMPT = """You are a helpful assistant that analyzes finance-related queries to determine which finance acts are relevant.

Given a user query and a list of available finance acts, identify which act(s) the query is most likely referring to.

Return ONLY a comma-separated list of act names (as they appear in the list), nothing else.
If the query could relate to multiple acts, include all relevant ones.
If the query is general and could relate to any act, return "all".
If no specific act is mentioned or relevant, return "all"."""

SYSTEM_PROMPT_NE = """तपाईं वित्त-सम्बन्धित प्रश्नहरू विश्लेषण गरेर कुन वित्त ऐनहरू सान्दर्भिक छन् भनी पहिचान गर्ने सहायक हुनुहुन्छ।

प्रयोगकर्ताको प्रश्न र उपलब्ध वित्त ऐनहरूको सूचीको आधारमा, कुन ऐन(हरू) सबैभन्दा सान्दर्भिक छन् पहिचान गर्नुहोस्।

केवल अल्पविराम-पृथक ऐनका नामहरू मात्र फर्काउनुहोस्, अरू केही होइन।
यदि प्रश्न सामान्य छ भने "all" फर्काउनुहोस्।"""

USER_PROMPT = """Available Finance Acts:
{act_list}

User Query: {user_query}

Which act(s) is this query about? Return only the act names separated by commas, or "all" if it's general."""

USER_PROMPT_NE = """उपलब्ध वित्त ऐनहरू:
{act_list}

प्रयोगकर्ताको प्रश्न: {user_query}

यो प्रश्न कुन ऐन(हरू)को बारेमा हो? केवल ऐनका नामहरू अल्पविरामले छुट्याएर फर्काउनुहोस्, वा "all" यदि सामान्य छ भने।"""


def get_prompts(language: str = "en") -> tuple[str, str]:
    """Return (system_prompt, user_prompt_template) for the given language."""
    if language == "ne":
        return SYSTEM_PROMPT_NE, USER_PROMPT_NE
    return SYSTEM_PROMPT, USER_PROMPT
