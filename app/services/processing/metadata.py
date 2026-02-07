"""
Metadata Extractor

Extracts structured metadata from finance act documents:
- Act Name (e.g., "VAT Act", "Income Tax Act")
- Section Numbers (e.g., "Section 12", "Section 13A")
- Year (e.g., "2081", "2024")
- Chapter/Part information
"""

import re
from typing import Any, Dict, List, Optional

from loguru import logger


class MetadataExtractor:
    """
    Extracts metadata from finance act documents.

    Uses pattern matching and heuristics to identify:
    - Act names
    - Section numbers
    - Years
    - Chapters and parts
    """

    def __init__(self):
        """Initialize the metadata extractor."""
        # Common finance act patterns (expanded list)
        self.act_patterns = [
            r"(?:Income\s+Tax\s+Act|Income\s+Tax)",
            r"(?:VAT\s+Act|Value\s+Added\s+Tax\s+Act|Value\s+Added\s+Tax)",
            r"(?:Finance\s+Act|Budget\s+Act)",
            r"(?:Company\s+Act|Companies\s+Act)",
            r"(?:Customs\s+Act|Customs)",
            r"(?:Excise\s+Duty\s+Act|Excise)",
            r"(?:Social\s+Security\s+Act|Social\s+Security)",
            r"(?:Banking\s+Act|Banking)",
            r"(?:Insurance\s+Act|Insurance)",
            r"(?:Securities\s+Act|Securities)",
            r"(?:Foreign\s+Exchange\s+Act|Foreign\s+Exchange)",
            r"(?:Money\s+Laundering\s+Act|Money\s+Laundering)",
            r"(?:Labor\s+Act|Labour\s+Act|Labor|Labour)",
        ]

        # Section number patterns
        self.section_pattern = r"(?:Section|Sec\.?|§)\s*(\d+[A-Z]?)"

        # Year patterns (Nepali year 2070-2090, English year 2010-2030)
        self.year_pattern = r"\b(?:20[0-9]{2}|20[6-9][0-9])\b"

    def extract_metadata(
        self, markdown_content: str, page_number: int = 0, document_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract metadata from document content.

        Args:
            markdown_content: Markdown document content
            page_number: Page number (for page-specific metadata)
            document_name: Optional document name/filename to help extract act name

        Returns:
            Dictionary with extracted metadata:
            {
                "act_name": str,
                "year": Optional[str],
                "sections": List[str],
                "chapters": List[str],
                "page": int,
            }
        """
        metadata = {
            "act_name": self._extract_act_name(markdown_content, document_name=document_name),
            "year": self._extract_year(markdown_content),
            "sections": self._extract_sections(markdown_content),
            "chapters": self._extract_chapters(markdown_content),
            "page": page_number,
        }

        return metadata

    def _extract_act_name(self, text: str, document_name: Optional[str] = None) -> Optional[str]:
        """
        Extract act name from document.

        Args:
            text: Document text
            document_name: Optional document name/filename to use as fallback

        Returns:
            Act name or None
        """
        # First, try to extract from document name if provided
        if document_name:
            # Clean document name (remove extension, path, etc.)
            clean_name = document_name
            # Remove file extension
            if "." in clean_name:
                clean_name = clean_name.rsplit(".", 1)[0]
            # Remove path
            if "/" in clean_name:
                clean_name = clean_name.split("/")[-1]
            if "\\" in clean_name:
                clean_name = clean_name.split("\\")[-1]
            
            logger.debug(f"Trying to extract act name from document name: {clean_name}")
            
            # Check if document name contains act patterns
            for pattern in self.act_patterns:
                match = re.search(pattern, clean_name, re.IGNORECASE)
                if match:
                    act_name = match.group(0).strip()
                    logger.info(f"Found act name from document name pattern: {act_name}")
                    return act_name
            
            # If document name contains "Act", use it as fallback
            if "act" in clean_name.lower():
                # Clean up the name: capitalize properly
                words = clean_name.replace("_", " ").replace("-", " ").split()
                # Capitalize each word
                clean_act_name = " ".join(word.capitalize() for word in words)
                logger.info(f"Extracted act name from document name: {clean_act_name}")
                return clean_act_name

        # Look for act names in the first 50 lines (expanded from 20)
        lines = text.split("\n")[:50]
        first_part = "\n".join(lines)

        for pattern in self.act_patterns:
            match = re.search(pattern, first_part, re.IGNORECASE)
            if match:
                return match.group(0).strip()

        # Fallback: look for "Act" with preceding capitalized words
        # More flexible pattern - allows for more words
        act_match = re.search(r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,5})\s+Act", first_part)
        if act_match:
            return act_match.group(0).strip()
        
        # Another fallback: look for any capitalized phrase ending with "Act"
        act_match2 = re.search(r"([A-Z][A-Za-z\s]{3,30}?)\s+Act", first_part)
        if act_match2:
            return act_match2.group(0).strip()

        return None

    def _extract_year(self, text: str) -> Optional[str]:
        """
        Extract year from document.

        Args:
            text: Document text

        Returns:
            Year string or None
        """
        # Look in first few lines for year
        lines = text.split("\n")[:30]
        first_part = "\n".join(lines)

        matches = re.findall(self.year_pattern, first_part)
        if matches:
            # Return the first year found (likely the act year)
            return matches[0]

        return None

    def _extract_sections(self, text: str) -> List[str]:
        """
        Extract all section numbers from document.

        Args:
            text: Document text

        Returns:
            List of section numbers (e.g., ["12", "13A", "14"])
        """
        matches = re.findall(self.section_pattern, text, re.IGNORECASE)
        # Remove duplicates and sort
        unique_sections = sorted(set(matches), key=lambda x: (len(x), x))
        return unique_sections

    def _extract_chapters(self, text: str) -> List[str]:
        """
        Extract chapter/part information.

        Args:
            text: Document text

        Returns:
            List of chapter/part identifiers
        """
        chapters = []

        # Look for "Chapter X" or "Part X"
        chapter_pattern = r"(?:Chapter|Part)\s+(\d+[A-Z]?)"
        matches = re.findall(chapter_pattern, text, re.IGNORECASE)
        chapters.extend(matches)

        return sorted(set(chapters))

    def attach_metadata_to_chunks(
        self, chunks: List[Dict[str, Any]], document_metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Attach document metadata to each chunk.

        Args:
            chunks: List of chunk dictionaries
            document_metadata: Document-level metadata

        Returns:
            Chunks with attached metadata
        """
        enriched_chunks = []

        for chunk in chunks:
            # Extract section from chunk text if present
            chunk_sections = self._extract_sections(chunk.get("text", ""))

            enriched_chunk = {
                **chunk,
                "metadata": {
                    **chunk.get("metadata", {}),
                    **document_metadata,
                    "sections_in_chunk": chunk_sections,
                },
            }
            enriched_chunks.append(enriched_chunk)

        return enriched_chunks
