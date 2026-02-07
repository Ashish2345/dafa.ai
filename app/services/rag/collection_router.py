"""
Collection Router

Uses LLM to determine which act collections to search based on user query.
"""

from typing import List, Optional

from loguru import logger

from app.services.llm import LLMService
from app.services.vector_store import VectorStoreService


class CollectionRouter:
    """
    Routes queries to appropriate act collections using LLM.

    Analyzes user queries to determine which finance act(s) are relevant,
    then returns the corresponding collection names to search.
    """

    def __init__(
        self,
        vector_store: Optional[VectorStoreService] = None,
        llm_service: Optional[LLMService] = None,
    ):
        """
        Initialize the collection router.

        Args:
            vector_store: Vector store service to get available collections
            llm_service: LLM service for query analysis
        """
        self.vector_store = vector_store or VectorStoreService()
        self.llm_service = llm_service or LLMService()

    def route_query(self, user_query: str) -> List[str]:
        """
        Determine which collections to search based on user query.

        Args:
            user_query: User's question/query

        Returns:
            List of collection names to search
        """
        logger.info(f"Routing query to collections: {user_query[:100]}...")

        # Get all available collections
        all_collections = self.vector_store.get_all_collections()
        
        if not all_collections:
            logger.warning("No collections available in Qdrant")
            return []

        # If only one collection, return it
        if len(all_collections) == 1:
            logger.info(f"Only one collection available: {all_collections[0]}")
            return all_collections

        # Use LLM to determine relevant collections
        try:
            relevant_collections = self._determine_collections_with_llm(user_query, all_collections)
            logger.info(f"LLM determined {len(relevant_collections)} relevant collections: {relevant_collections}")
            return relevant_collections
        except Exception as e:
            logger.error(f"Error routing query with LLM: {e}")
            # Fallback: return all collections if LLM fails
            logger.warning("Falling back to searching all collections")
            return all_collections

    def _determine_collections_with_llm(
        self, user_query: str, available_collections: List[str]
    ) -> List[str]:
        """
        Use LLM to determine which collections are relevant to the query.

        Args:
            user_query: User's question/query
            available_collections: List of all available collection names

        Returns:
            List of relevant collection names
        """
        # Extract act names from collection names
        # Collection names are typically like "vat_act", "income_tax_act", etc.
        act_names = [self._collection_to_act_name(col) for col in available_collections]
        act_list = "\n".join([f"- {act}" for act in act_names if act])

        system_instruction = """You are a helpful assistant that analyzes finance-related queries to determine which finance acts are relevant.

Given a user query and a list of available finance acts, identify which act(s) the query is most likely referring to.

Return ONLY a comma-separated list of act names (as they appear in the list), nothing else.
If the query could relate to multiple acts, include all relevant ones.
If the query is general and could relate to any act, return "all".
If no specific act is mentioned or relevant, return "all"."""

        user_prompt = f"""Available Finance Acts:
{act_list}

User Query: {user_query}

Which act(s) is this query about? Return only the act names separated by commas, or "all" if it's general."""

        try:
            response = self.llm_service.call(
                prompt=user_prompt,
                system_instruction=system_instruction,
                temperature=0.1,  # Low temperature for deterministic routing
                max_tokens=200,
            )

            # Parse response
            response = response.strip().lower()
            
            if "all" in response:
                logger.debug("LLM determined query is general, returning all collections")
                return available_collections

            # Extract act names from response
            relevant_acts = [act.strip() for act in response.split(",")]
            
            # Map act names back to collection names
            relevant_collections = []
            for act in relevant_acts:
                # Find matching collection
                for collection in available_collections:
                    act_name = self._collection_to_act_name(collection)
                    if act_name and act.lower() in act_name.lower():
                        relevant_collections.append(collection)
                        break

            # If no matches found, return all collections as fallback
            if not relevant_collections:
                logger.warning(f"Could not map LLM response '{response}' to collections, using all")
                return available_collections

            return list(set(relevant_collections))  # Remove duplicates

        except Exception as e:
            logger.error(f"Error in LLM collection routing: {e}")
            return available_collections

    def _collection_to_act_name(self, collection_name: str) -> Optional[str]:
        """
        Convert collection name to human-readable act name.

        Args:
            collection_name: Collection name (e.g., "vat_act", "income_tax_act")

        Returns:
            Act name (e.g., "VAT Act", "Income Tax Act")
        """
        # Remove common prefixes/suffixes
        name = collection_name.lower()
        
        # Replace underscores with spaces and title case
        name = name.replace("_", " ").title()
        
        # Common mappings
        mappings = {
            "Vat Act": "VAT Act",
            "Income Tax Act": "Income Tax Act",
            "Finance Act": "Finance Act",
            "Customs Act": "Customs Act",
            "Company Act": "Company Act",
            "Companies Act": "Companies Act",
        }
        
        for key, value in mappings.items():
            if key.lower() in name.lower():
                return value
        
        return name
