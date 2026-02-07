"""
Query Enhancement Service

Enhances user queries for better retrieval:
- Query expansion (synonyms, related terms)
- Entity extraction (section numbers, act names)
- Query rewriting
- Intent classification
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger


class QueryEnhancer:
    """
    Enhances queries for better retrieval accuracy.
    
    Provides query expansion, entity extraction, and query rewriting
    to improve both vector and keyword search results.
    """

    def __init__(self):
        """Initialize the query enhancer."""
        # Legal term synonyms and expansions
        self.legal_synonyms = {
            "tax": ["taxation", "levy", "duty", "assessment"],
            "income": ["earnings", "revenue", "salary", "wages"],
            "rate": ["percentage", "proportion", "ratio"],
            "section": ["sec", "sec.", "§"],
            "act": ["law", "legislation", "statute"],
            "monthly": ["per month", "monthly", "each month"],
            "annual": ["yearly", "per year", "annually"],
        }

        # Finance-specific terms
        self.finance_terms = {
            "vat": ["value added tax", "sales tax"],
            "income tax": ["personal income tax", "individual tax"],
            "taxable": ["assessable", "chargeable"],
            "deduction": ["allowance", "exemption", "rebate"],
        }

    def enhance_query(
        self,
        query: str,
        expand: bool = True,
        extract_entities: bool = True,
    ) -> Dict[str, Any]:
        """
        Enhance a user query for better retrieval.
        
        Args:
            query: Original user query
            expand: Whether to expand query with synonyms
            extract_entities: Whether to extract entities (section numbers, etc.)
            
        Returns:
            Dictionary with enhanced query information:
            {
                "original": str,
                "enhanced": str,
                "expanded_terms": List[str],
                "entities": Dict,
                "keywords": List[str],
            }
        """
        enhanced_query = query
        expanded_terms = []
        entities = {}
        keywords = []

        # Extract entities
        if extract_entities:
            entities = self._extract_entities(query)

        # Expand query with synonyms
        if expand:
            enhanced_query, expanded_terms = self._expand_query(query)

        # Extract keywords
        keywords = self._extract_keywords(enhanced_query)

        return {
            "original": query,
            "enhanced": enhanced_query,
            "expanded_terms": expanded_terms,
            "entities": entities,
            "keywords": keywords,
        }

    def _extract_entities(self, query: str) -> Dict[str, Any]:
        """
        Extract entities from query (section numbers, act names, etc.).
        
        Args:
            query: User query
            
        Returns:
            Dictionary with extracted entities:
            {
                "sections": List[str],
                "act_names": List[str],
                "numbers": List[str],
                "years": List[str],
            }
        """
        entities = {
            "sections": [],
            "act_names": [],
            "numbers": [],
            "years": [],
        }

        # Extract section numbers
        section_pattern = r"(?:section|sec\.?|§)\s*(\d+[a-z]?)"
        sections = re.findall(section_pattern, query, re.IGNORECASE)
        entities["sections"] = [s.lower() for s in sections]

        # Extract act names
        act_patterns = [
            r"(?:vat|value\s+added\s+tax)\s+act",
            r"income\s+tax\s+act",
            r"finance\s+act",
            r"customs\s+act",
            r"company\s+act",
        ]
        for pattern in act_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            entities["act_names"].extend([m.lower() for m in matches])

        # Extract numbers (for calculations, amounts, etc.)
        number_pattern = r'\b(\d+(?:,\d{3})*(?:\.\d+)?)\b'  # Numbers with commas/decimals
        numbers = re.findall(number_pattern, query)
        entities["numbers"] = numbers

        # Extract years
        year_pattern = r'\b(20\d{2})\b'  # Years 2000-2099
        years = re.findall(year_pattern, query)
        entities["years"] = years

        return entities

    def _expand_query(self, query: str) -> Tuple[str, List[str]]:
        """
        Expand query with synonyms and related terms.
        
        Args:
            query: Original query
            
        Returns:
            Tuple of (expanded_query, expanded_terms)
        """
        expanded_terms = []
        expanded_query = query

        # Convert to lowercase for matching
        query_lower = query.lower()

        # Find and expand terms
        for term, synonyms in self.legal_synonyms.items():
            if term in query_lower:
                # Add synonyms to expanded terms
                expanded_terms.extend(synonyms)
                # Optionally add to query (for now, just track)

        # Finance-specific expansions
        for term, expansions in self.finance_terms.items():
            if term in query_lower:
                expanded_terms.extend(expansions)

        # Build expanded query (original + expanded terms)
        if expanded_terms:
            # Add expanded terms to query
            expanded_query = f"{query} {' '.join(expanded_terms[:5])}"  # Limit to 5 terms

        return expanded_query, expanded_terms

    def _extract_keywords(self, query: str) -> List[str]:
        """
        Extract important keywords from query.
        
        Args:
            query: Query string
            
        Returns:
            List of important keywords
        """
        # Remove stop words (simple list for finance domain)
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "should",
            "could", "may", "might", "must", "can", "this", "that", "these", "those",
            "i", "you", "he", "she", "it", "we", "they", "my", "your", "his", "her",
            "its", "our", "their", "what", "which", "who", "whom", "whose", "where",
            "when", "why", "how", "if", "then", "than", "as", "so",
        }

        # Tokenize and filter
        words = re.findall(r'\b[a-z]+\b', query.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 2]

        # Also extract special patterns (section numbers, etc.)
        special_patterns = [
            r'section\s+\d+[a-z]?',
            r'chapter\s+\d+',
            r'\d+[a-z]?',  # Numbers
        ]

        for pattern in special_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            keywords.extend([m.lower() for m in matches])

        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                unique_keywords.append(kw)

        return unique_keywords

    def classify_intent(self, query: str) -> str:
        """
        Classify query intent for better retrieval strategy.
        
        Args:
            query: User query
            
        Returns:
            Intent type: "factual", "calculation", "comparison", "definition", "general"
        """
        query_lower = query.lower()

        # Calculation intent (contains numbers and calculation words)
        calculation_words = ["calculate", "compute", "how much", "what is", "rate", "percentage"]
        if any(word in query_lower for word in calculation_words) and re.search(r'\d+', query):
            return "calculation"

        # Definition intent
        definition_words = ["what is", "define", "meaning", "means"]
        if any(word in query_lower for word in definition_words):
            return "definition"

        # Comparison intent
        comparison_words = ["compare", "difference", "vs", "versus", "between"]
        if any(word in query_lower for word in comparison_words):
            return "comparison"

        # Factual lookup (section-specific)
        if re.search(r'section\s+\d+', query_lower):
            return "factual"

        return "general"
