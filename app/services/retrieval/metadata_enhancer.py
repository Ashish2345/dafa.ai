"""
Metadata-Enhanced Retrieval

Enhances retrieval by leveraging document metadata:
- Metadata-based filtering
- Metadata-based relevance boosting
- Query-to-metadata matching
- Section/Act-specific retrieval
"""

import re
from typing import Any, Dict, List, Optional

from loguru import logger


class MetadataEnhancer:
    """
    Enhances retrieval using document metadata.
    
    Provides metadata-based filtering, boosting, and query matching
    to improve retrieval accuracy for finance act documents.
    """

    def __init__(self):
        """Initialize the metadata enhancer."""
        # Metadata fields we care about
        self.metadata_fields = [
            "act_name",
            "sections_in_chunk",
            "year",
            "chapters",
            "sections",
            "document_name",
        ]

    def extract_query_metadata(self, query: str) -> Dict[str, Any]:
        """
        Extract metadata from user query for filtering/boosting.
        
        Args:
            query: User query string
            
        Returns:
            Dictionary with extracted metadata:
            {
                "act_name": Optional[str],
                "sections": List[str],
                "year": Optional[str],
                "chapters": List[str],
            }
        """
        metadata = {
            "act_name": None,
            "sections": [],
            "year": None,
            "chapters": [],
        }

        query_lower = query.lower()

        # Extract act names
        act_patterns = [
            (r"(?:vat|value\s+added\s+tax)\s+act", "VAT Act"),
            (r"income\s+tax\s+act", "Income Tax Act"),
            (r"finance\s+act", "Finance Act"),
            (r"customs\s+act", "Customs Act"),
            (r"company\s+act", "Company Act"),
            (r"excise\s+duty\s+act", "Excise Duty Act"),
            (r"(?:labor|labour)\s+act", "Labor Act"),
        ]

        for pattern, act_name in act_patterns:
            if re.search(pattern, query_lower):
                metadata["act_name"] = act_name
                break

        # Extract section numbers
        section_pattern = r"(?:section|sec\.?|§)\s*(\d+[a-z]?)"
        sections = re.findall(section_pattern, query_lower)
        metadata["sections"] = [s.lower() for s in sections]

        # Extract year
        year_pattern = r'\b(20\d{2})\b'
        years = re.findall(year_pattern, query)
        if years:
            metadata["year"] = years[0]  # Use first year found

        # Extract chapters/parts
        chapter_pattern = r"(?:chapter|part)\s+(\d+[a-z]?)"
        chapters = re.findall(chapter_pattern, query_lower)
        metadata["chapters"] = [c.lower() for c in chapters]

        return metadata

    def build_metadata_filter(
        self,
        query_metadata: Dict[str, Any],
        strict: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        Build Qdrant filter conditions from query metadata.
        
        Args:
            query_metadata: Extracted metadata from query
            strict: If True, all conditions must match (AND). If False, any can match (OR).
            
        Returns:
            Filter conditions dictionary for Qdrant, or None if no filters
        """
        filters = {}

        # Act name filter (exact match)
        if query_metadata.get("act_name"):
            filters["act_name"] = query_metadata["act_name"]

        # Year filter (exact match)
        if query_metadata.get("year"):
            filters["year"] = query_metadata["year"]

        # Section filter (if specific section requested)
        if query_metadata.get("sections"):
            # For sections, we'll use boosting instead of strict filtering
            # (chunks may contain multiple sections)
            pass

        # Chapter filter
        if query_metadata.get("chapters"):
            # Similar to sections, use boosting
            pass

        return filters if filters else None

    def boost_by_metadata(
        self,
        results: List[Dict[str, Any]],
        query_metadata: Dict[str, Any],
        boost_weight: float = 0.2,
    ) -> List[Dict[str, Any]]:
        """
        Boost result scores based on metadata matches.
        
        Args:
            results: List of search results with metadata
            query_metadata: Extracted metadata from query
            boost_weight: Weight for metadata boosting (0.0-1.0)
            
        Returns:
            Results with boosted scores
        """
        if not query_metadata or not results:
            return results

        boosted_results = []

        for result in results:
            result_metadata = result.get("metadata", {})
            boost = 0.0

            # Boost for act name match
            query_act = query_metadata.get("act_name")
            if query_act:
                result_act = result_metadata.get("act_name", "")
                if query_act.lower() in result_act.lower() or result_act.lower() in query_act.lower():
                    boost += 0.3

            # Boost for section matches
            query_sections = query_metadata.get("sections", [])
            if query_sections:
                result_sections = result_metadata.get("sections_in_chunk", [])
                # Check if any query section is in result sections
                matching_sections = set(query_sections) & set(
                    [s.lower() for s in result_sections]
                )
                if matching_sections:
                    # More matches = higher boost
                    boost += 0.4 * (len(matching_sections) / len(query_sections))

            # Boost for year match
            query_year = query_metadata.get("year")
            if query_year:
                result_year = result_metadata.get("year", "")
                if query_year == result_year:
                    boost += 0.2

            # Boost for chapter matches
            query_chapters = query_metadata.get("chapters", [])
            if query_chapters:
                result_chapters = result_metadata.get("chapters", [])
                matching_chapters = set(query_chapters) & set(
                    [c.lower() for c in result_chapters]
                )
                if matching_chapters:
                    boost += 0.1 * (len(matching_chapters) / len(query_chapters))

            # Apply boost to score
            original_score = result.get("score", 0.0)
            boosted_score = original_score + (boost * boost_weight)
            result["score"] = boosted_score
            result["metadata_boost"] = boost  # Track boost for debugging

            boosted_results.append(result)

        # Re-sort by boosted score
        boosted_results.sort(key=lambda x: x.get("score", 0.0), reverse=True)

        logger.debug(
            f"Applied metadata boosting to {len(boosted_results)} results "
            f"(avg boost: {sum(r.get('metadata_boost', 0) for r in boosted_results) / len(boosted_results) if boosted_results else 0:.3f})"
        )

        return boosted_results

    def prioritize_by_metadata(
        self,
        results: List[Dict[str, Any]],
        query_metadata: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Prioritize results that match query metadata.
        
        This is a more aggressive version of boosting that re-orders
        results based on metadata matches.
        
        Args:
            results: List of search results
            query_metadata: Extracted metadata from query
            
        Returns:
            Re-ordered results with metadata matches first
        """
        if not query_metadata or not results:
            return results

        # Separate results into matched and unmatched
        matched = []
        unmatched = []

        for result in results:
            result_metadata = result.get("metadata", {})
            matches = False

            # Check act name match
            query_act = query_metadata.get("act_name")
            if query_act:
                result_act = result_metadata.get("act_name", "")
                if query_act.lower() in result_act.lower() or result_act.lower() in query_act.lower():
                    matches = True

            # Check section matches
            query_sections = query_metadata.get("sections", [])
            if query_sections:
                result_sections = result_metadata.get("sections_in_chunk", [])
                if any(s.lower() in [rs.lower() for rs in result_sections] for s in query_sections):
                    matches = True

            # Check year match
            query_year = query_metadata.get("year")
            if query_year:
                result_year = result_metadata.get("year", "")
                if query_year == result_year:
                    matches = True

            if matches:
                matched.append(result)
            else:
                unmatched.append(result)

        # Combine: matched first, then unmatched (both sorted by score)
        matched.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        unmatched.sort(key=lambda x: x.get("score", 0.0), reverse=True)

        prioritized = matched + unmatched

        logger.debug(
            f"Prioritized {len(matched)} metadata-matched results out of {len(results)} total"
        )

        return prioritized

    def get_metadata_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get summary of metadata from search results.
        
        Useful for understanding what acts/sections were retrieved.
        
        Args:
            results: List of search results
            
        Returns:
            Summary dictionary with metadata statistics
        """
        summary = {
            "acts": {},
            "sections": set(),
            "years": set(),
            "total_results": len(results),
        }

        for result in results:
            metadata = result.get("metadata", {})

            # Count acts
            act_name = metadata.get("act_name")
            if act_name:
                summary["acts"][act_name] = summary["acts"].get(act_name, 0) + 1

            # Collect sections
            sections = metadata.get("sections_in_chunk", [])
            summary["sections"].update(sections)

            # Collect years
            year = metadata.get("year")
            if year:
                summary["years"].add(year)

        # Convert sets to lists for JSON serialization
        summary["sections"] = sorted(list(summary["sections"]))
        summary["years"] = sorted(list(summary["years"]))

        return summary
