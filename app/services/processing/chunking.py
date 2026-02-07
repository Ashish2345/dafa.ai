"""
Chunking Service

Handles text chunking for RAG ingestion using LlamaIndex SentenceSplitter.
Maintains section-aware chunking for finance act documents with enhanced
legal document structure awareness and quality scoring.
"""

import re
from typing import Any, Dict, List

from llama_index.core.node_parser import SentenceSplitter
from loguru import logger


class ChunkingService:
    """
    Service for chunking documents into smaller pieces for RAG.

    Uses LlamaIndex SentenceSplitter for industry-standard chunking
    while maintaining section-aware chunking for finance documents.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        respect_sections: bool = True,
        preserve_legal_concepts: bool = True,
        min_chunk_size: int = 100,
        max_chunk_size: int = 2000,
    ):
        """
        Initialize the chunking service.

        Args:
            chunk_size: Target size of each chunk in characters
            chunk_overlap: Number of characters to overlap between chunks
            respect_sections: Whether to respect section boundaries when chunking
            preserve_legal_concepts: Whether to preserve complete legal concepts (definitions, clauses)
            min_chunk_size: Minimum chunk size (chunks smaller than this will be merged)
            max_chunk_size: Maximum chunk size (chunks larger than this will be split)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.respect_sections = respect_sections
        self.preserve_legal_concepts = preserve_legal_concepts
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

        # Initialize LlamaIndex SentenceSplitter
        self.splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator=" ",  # Split on spaces for better word boundaries
        )
        
        # Legal document patterns for better boundary detection
        self._init_legal_patterns()

    def chunk_document(self, markdown_content: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Chunk a Markdown document into smaller pieces using LlamaIndex.

        Args:
            markdown_content: Complete Markdown document
            metadata: Optional document-level metadata

        Returns:
            List of chunk dictionaries:
            [
                {
                    "chunk_id": str,
                    "text": str,
                    "metadata": Dict,
                    "start_char": int,
                    "end_char": int,
                }
            ]
        """
        if not markdown_content:
            return []

        logger.info(f"Chunking document (size: {len(markdown_content)} chars) using LlamaIndex")

        doc_metadata = metadata or {}

        # If section-aware chunking is enabled, split by sections first
        if self.respect_sections:
            chunks = self._chunk_with_sections(markdown_content, doc_metadata)
        else:
            chunks = self._chunk_with_llamaindex(markdown_content, doc_metadata)

        # Post-process chunks: preserve legal concepts and score quality
        if self.preserve_legal_concepts:
            chunks = self._preserve_legal_concepts(chunks)
        
        # Score chunk quality
        chunks = self._score_chunk_quality(chunks)
        
        # Merge very small chunks and split very large ones
        chunks = self._optimize_chunk_sizes(chunks)
        
        logger.info(
            f"Created {len(chunks)} chunks using LlamaIndex SentenceSplitter "
            f"(avg quality: {sum(c.get('quality_score', 0) for c in chunks) / len(chunks) if chunks else 0:.2f})"
        )
        return chunks

    def _chunk_with_sections(self, text: str, doc_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Chunk document with section awareness.

        First splits by sections, then uses LlamaIndex to chunk each section.

        Args:
            text: Text to chunk
            doc_metadata: Document metadata

        Returns:
            List of chunks
        """
        sections = self._split_by_sections(text)
        all_chunks = []
        chunk_id = 0
        current_offset = 0

        for section_text in sections:
            if not section_text.strip():
                continue

            # Use LlamaIndex to chunk this section
            section_chunks = self._chunk_with_llamaindex(section_text, doc_metadata, chunk_id, current_offset)
            all_chunks.extend(section_chunks)

            # Update offsets for next section
            if section_chunks:
                chunk_id = len(all_chunks)
                # Update offset based on the last chunk's end position
                if section_chunks:
                    current_offset = section_chunks[-1]["end_char"]
                else:
                    current_offset += len(section_text)

        return all_chunks

    def _chunk_with_llamaindex(
        self,
        text: str,
        doc_metadata: Dict[str, Any],
        chunk_id_start: int = 0,
        start_offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Chunk text using LlamaIndex SentenceSplitter.

        Args:
            text: Text to chunk
            doc_metadata: Document metadata
            chunk_id_start: Starting chunk ID
            start_offset: Character offset for this section

        Returns:
            List of chunks in our format
        """
        try:
            # Use LlamaIndex SentenceSplitter to create nodes
            from llama_index.core.schema import Document

            doc = Document(text=text)
            nodes = self.splitter.get_nodes_from_documents([doc])

            chunks = []
            current_pos = start_offset

            for idx, node in enumerate(nodes):
                chunk_id = chunk_id_start + idx
                node_text = node.get_content()

                # Calculate character positions
                # For section-aware chunking, we track position relative to the full document
                start_char = current_pos
                end_char = current_pos + len(node_text)

                chunks.append({
                    "chunk_id": f"chunk_{chunk_id}",
                    "text": node_text,
                    "metadata": {
                        **doc_metadata,
                        "start_char": start_char,
                        "end_char": end_char,
                        "chunk_index": chunk_id,
                    },
                    "start_char": start_char,
                    "end_char": end_char,
                })

                # Update position for next chunk (accounting for overlap)
                # LlamaIndex handles overlap internally, so we just move forward
                current_pos = end_char

            return chunks

        except Exception as e:
            logger.warning(f"Error using LlamaIndex chunking, falling back to basic chunking: {e}")
            # Fallback to basic chunking if LlamaIndex fails
            return self._basic_chunking_fallback(text, doc_metadata, chunk_id_start, start_offset)

    def _basic_chunking_fallback(
        self, text: str, doc_metadata: Dict[str, Any], chunk_id_start: int, start_offset: int
    ) -> List[Dict[str, Any]]:
        """
        Fallback basic chunking if LlamaIndex fails.

        Args:
            text: Text to chunk
            doc_metadata: Document metadata
            chunk_id_start: Starting chunk ID
            start_offset: Character offset

        Returns:
            List of chunks
        """
        chunks = []
        text_length = len(text)
        current_pos = 0
        chunk_id = chunk_id_start

        while current_pos < text_length:
            end_pos = min(current_pos + self.chunk_size, text_length)

            # Try to break at sentence boundary
            if end_pos < text_length:
                sentence_end = self._find_sentence_boundary(text, current_pos, end_pos)
                if sentence_end > current_pos:
                    end_pos = sentence_end

            chunk_text = text[current_pos:end_pos].strip()

            if chunk_text:
                chunks.append({
                    "chunk_id": f"chunk_{chunk_id}",
                    "text": chunk_text,
                    "metadata": {
                        **doc_metadata,
                        "start_char": start_offset + current_pos,
                        "end_char": start_offset + end_pos,
                        "chunk_index": chunk_id,
                    },
                    "start_char": start_offset + current_pos,
                    "end_char": start_offset + end_pos,
                })
                chunk_id += 1

            current_pos = end_pos - self.chunk_overlap
            if current_pos <= 0:
                current_pos = end_pos

        return chunks

    def _init_legal_patterns(self):
        """Initialize patterns for legal document structure detection."""
        # Section patterns (main sections, subsections, clauses)
        self.section_patterns = [
            r"(?=\n(?:Section|Sec\.?|§)\s+\d+[A-Z]?)",  # Section 12, Section 12A
            r"(?=\n(?:Sub-section|Subsection|Sub-sec\.?)\s+\d+)",  # Sub-section 1
            r"(?=\n(?:Clause|Cl\.?)\s+\([a-z0-9]+\))",  # Clause (a), Clause (1)
            r"(?=\n(?:Sub-clause|Subclause)\s+\([a-z0-9]+\))",  # Sub-clause (i)
        ]
        
        # Chapter/Part patterns
        self.chapter_patterns = [
            r"(?=\n(?:Chapter|Ch\.?)\s+\d+)",
            r"(?=\n(?:Part|Pt\.?)\s+\d+)",
            r"(?=\n(?:Article|Art\.?)\s+\d+)",
            r"(?=\n(?:Schedule|Sch\.?)\s+\d+)",
        ]
        
        # Definition patterns (legal definitions often span multiple sentences)
        self.definition_patterns = [
            r'"(?:[^"]+)"\s+means\s+',  # "term" means ...
            r'"(?:[^"]+)"\s+shall\s+mean\s+',  # "term" shall mean ...
            r'means\s+"(?:[^"]+)"',  # means "term"
            r'shall\s+mean\s+"(?:[^"]+)"',  # shall mean "term"
        ]
        
        # Table patterns
        self.table_pattern = r"(?=\n\|.*\|.*\n)"  # Markdown tables
        
        # List patterns (numbered lists, bullet lists)
        self.list_patterns = [
            r"(?=\n\s*\d+[\.\)]\s+)",  # Numbered list: 1. or 1)
            r"(?=\n\s*[a-z][\.\)]\s+)",  # Lettered list: a. or a)
            r"(?=\n\s*[-*•]\s+)",  # Bullet list
        ]

    def _split_by_sections(self, text: str) -> List[str]:
        """
        Split text by section boundaries with enhanced legal document awareness.

        Detects:
        - Main sections (Section 12, Section 12A)
        - Subsections and clauses
        - Chapters and Parts
        - Legal definitions
        - Tables and structured content

        Args:
            text: Text to split

        Returns:
            List of section texts
        """
        # First, split by major boundaries (Chapters, Parts)
        sections = [text]
        for pattern in self.chapter_patterns:
            new_sections = []
            for section in sections:
                split_sections = re.split(pattern, section)
                new_sections.extend(split_sections)
            sections = new_sections

        # Then split by sections within each chapter/part
        all_sections = []
        for section in sections:
            section_parts = self._split_by_section_markers(section)
            all_sections.extend(section_parts)

        # Also split by Markdown headers
        final_sections = []
        header_pattern = r"(?=\n#{1,3}\s+)"
        for section in all_sections:
            header_sections = re.split(header_pattern, section)
            final_sections.extend(header_sections)

        # Filter empty sections and preserve legal definitions
        result = []
        for section in final_sections:
            section = section.strip()
            if section:
                # Check if this section contains a definition that should be kept together
                if self._contains_definition(section):
                    result.append(section)
                else:
                    result.append(section)

        return result

    def _split_by_section_markers(self, text: str) -> List[str]:
        """
        Split text by section markers (Section, Sub-section, Clause, etc.).

        Args:
            text: Text to split

        Returns:
            List of section texts
        """
        sections = [text]
        
        # Split by section patterns (in order of hierarchy)
        for pattern in self.section_patterns:
            new_sections = []
            for section in sections:
                split_sections = re.split(pattern, section)
                new_sections.extend(split_sections)
            sections = new_sections
        
        return [s.strip() for s in sections if s.strip()]

    def _contains_definition(self, text: str) -> bool:
        """
        Check if text contains a legal definition pattern.

        Args:
            text: Text to check

        Returns:
            True if text contains a definition pattern
        """
        for pattern in self.definition_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def _preserve_legal_concepts(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Preserve complete legal concepts by merging chunks that contain incomplete definitions or clauses.

        Args:
            chunks: List of chunks

        Returns:
            List of chunks with legal concepts preserved
        """
        if not chunks:
            return chunks

        merged_chunks = []
        i = 0

        while i < len(chunks):
            current_chunk = chunks[i].copy()
            current_text = current_chunk.get("text", "")
            
            # Check if current chunk contains start of a definition
            if self._is_incomplete_definition(current_text):
                # Try to merge with next chunk(s) until definition is complete
                merged_text = current_text
                j = i + 1
                
                while j < len(chunks) and len(merged_text) < self.max_chunk_size:
                    next_text = chunks[j].get("text", "")
                    merged_text += " " + next_text
                    
                    # Check if definition is now complete
                    if self._is_complete_definition(merged_text):
                        # Merge chunks
                        current_chunk["text"] = merged_text
                        current_chunk["end_char"] = chunks[j]["end_char"]
                        current_chunk["metadata"]["merged_chunks"] = j - i + 1
                        i = j + 1
                        break
                    j += 1
                else:
                    # Definition still incomplete, but reached max size or end
                    if j < len(chunks):
                        current_chunk["text"] = merged_text
                        current_chunk["end_char"] = chunks[j - 1]["end_char"]
                        current_chunk["metadata"]["merged_chunks"] = j - i
                        i = j
                    else:
                        i += 1
            else:
                i += 1
            
            merged_chunks.append(current_chunk)

        return merged_chunks

    def _is_incomplete_definition(self, text: str) -> bool:
        """
        Check if text contains an incomplete definition (starts but doesn't end properly).

        Args:
            text: Text to check

        Returns:
            True if definition appears incomplete
        """
        # Check for definition start patterns
        has_definition_start = any(
            re.search(pattern, text, re.IGNORECASE) for pattern in self.definition_patterns
        )
        
        if not has_definition_start:
            return False
        
        # Check if definition seems complete (ends with proper punctuation)
        # Legal definitions often end with semicolon, period, or new section
        definition_end_patterns = [
            r'[;.]\s*(?:\n|$)',  # Ends with semicolon/period
            r'[;.]\s*(?:Section|Chapter|Part|Article)',  # Followed by new section
        ]
        
        has_complete_end = any(
            re.search(pattern, text, re.IGNORECASE) for pattern in definition_end_patterns
        )
        
        return not has_complete_end

    def _is_complete_definition(self, text: str) -> bool:
        """
        Check if text contains a complete definition.

        Args:
            text: Text to check

        Returns:
            True if definition appears complete
        """
        if not self._contains_definition(text):
            return False
        
        # Check for proper ending
        definition_end_patterns = [
            r'[;.]\s*(?:\n|$)',  # Ends with semicolon/period
            r'[;.]\s*(?:Section|Chapter|Part|Article)',  # Followed by new section
            r'[;.]\s*(?:\([a-z]\)|\([0-9]\))',  # Followed by clause marker
        ]
        
        return any(
            re.search(pattern, text, re.IGNORECASE) for pattern in definition_end_patterns
        )

    def _score_chunk_quality(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Score chunk quality based on completeness, coherence, and metadata richness.

        Args:
            chunks: List of chunks

        Returns:
            List of chunks with quality scores added
        """
        scored_chunks = []
        
        for chunk in chunks:
            text = chunk.get("text", "")
            metadata = chunk.get("metadata", {})
            
            # Calculate quality metrics
            completeness_score = self._score_completeness(text)
            coherence_score = self._score_coherence(text)
            metadata_score = self._score_metadata_richness(metadata)
            
            # Weighted quality score (0.0 to 1.0)
            quality_score = (
                0.4 * completeness_score +
                0.3 * coherence_score +
                0.3 * metadata_score
            )
            
            # Add quality scores to chunk
            chunk["quality_score"] = round(quality_score, 3)
            chunk["metadata"]["quality_metrics"] = {
                "completeness": round(completeness_score, 3),
                "coherence": round(coherence_score, 3),
                "metadata_richness": round(metadata_score, 3),
            }
            
            scored_chunks.append(chunk)
        
        return scored_chunks

    def _score_completeness(self, text: str) -> float:
        """
        Score chunk completeness (doesn't cut mid-sentence/clause).

        Args:
            text: Chunk text

        Returns:
            Completeness score (0.0 to 1.0)
        """
        if not text:
            return 0.0
        
        score = 1.0
        
        # Check if ends with proper punctuation
        text_stripped = text.rstrip()
        if not text_stripped:
            return 0.0
        
        # Proper endings: period, exclamation, question mark, semicolon, colon
        proper_endings = ".!?;:"
        if text_stripped[-1] not in proper_endings:
            score -= 0.3  # Penalize for incomplete sentence
        
        # Check for incomplete parentheses, brackets, quotes
        open_parens = text.count("(") - text.count(")")
        open_brackets = text.count("[") - text.count("]")
        open_braces = text.count("{") - text.count("}")
        open_quotes = text.count('"') % 2
        
        if open_parens > 0:
            score -= 0.2
        if open_brackets > 0:
            score -= 0.1
        if open_braces > 0:
            score -= 0.1
        if open_quotes > 0:
            score -= 0.1
        
        # Check for incomplete list items
        if re.search(r'\n\s*\d+[\.\)]\s*$', text):  # Ends with numbered list start
            score -= 0.2
        
        return max(0.0, min(1.0, score))

    def _score_coherence(self, text: str) -> float:
        """
        Score chunk coherence (related concepts together, logical flow).

        Args:
            text: Chunk text

        Returns:
            Coherence score (0.0 to 1.0)
        """
        if not text or len(text) < 50:
            return 0.5  # Small chunks get neutral score
        
        score = 1.0
        
        # Check for topic consistency (simple heuristic: repeated key terms)
        words = text.lower().split()
        if len(words) < 10:
            return 0.5
        
        # Count unique words vs total words (lower ratio = more repetition = more coherent)
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio > 0.9:  # Very high uniqueness (might be incoherent)
            score -= 0.2
        elif unique_ratio < 0.6:  # Good repetition (coherent)
            score += 0.1
        
        # Check for legal term consistency (section numbers, act names)
        section_mentions = len(re.findall(r'Section\s+\d+', text, re.IGNORECASE))
        if section_mentions > 1:
            score += 0.1  # Multiple section mentions suggest coherence
        
        # Check for abrupt topic changes (heuristic: many different section numbers)
        all_sections = re.findall(r'Section\s+(\d+)', text, re.IGNORECASE)
        if len(set(all_sections)) > 3:  # Too many different sections
            score -= 0.2
        
        return max(0.0, min(1.0, score))

    def _score_metadata_richness(self, metadata: Dict[str, Any]) -> float:
        """
        Score metadata richness (has section numbers, act name, etc.).

        Args:
            metadata: Chunk metadata

        Returns:
            Metadata richness score (0.0 to 1.0)
        """
        score = 0.0
        
        # Act name (important)
        if metadata.get("act_name"):
            score += 0.3
        
        # Section numbers (very important)
        sections = metadata.get("sections_in_chunk", [])
        if sections:
            score += 0.4
            if len(sections) > 1:
                score += 0.1  # Bonus for multiple sections
        
        # Chapter/Part information
        if metadata.get("chapters"):
            score += 0.1
        
        # Year
        if metadata.get("year"):
            score += 0.1
        
        return min(1.0, score)

    def _optimize_chunk_sizes(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Optimize chunk sizes by merging very small chunks and splitting very large ones.

        Args:
            chunks: List of chunks

        Returns:
            List of optimized chunks
        """
        if not chunks:
            return chunks

        optimized = []
        i = 0

        while i < len(chunks):
            current_chunk = chunks[i].copy()
            current_text = current_chunk.get("text", "")
            current_size = len(current_text)

            # If chunk is too small, try to merge with next chunk
            if current_size < self.min_chunk_size and i + 1 < len(chunks):
                next_chunk = chunks[i + 1]
                next_text = next_chunk.get("text", "")
                merged_size = current_size + len(next_text) + 1  # +1 for space

                # Merge if combined size is reasonable
                if merged_size <= self.max_chunk_size:
                    current_chunk["text"] = current_text + " " + next_text
                    current_chunk["end_char"] = next_chunk["end_char"]
                    current_chunk["metadata"]["merged_small_chunks"] = True
                    i += 2  # Skip next chunk as it's merged
                else:
                    i += 1
            # If chunk is too large, split it
            elif current_size > self.max_chunk_size:
                split_chunks = self._split_large_chunk(current_chunk)
                optimized.extend(split_chunks)
                i += 1
            else:
                i += 1

            if current_chunk.get("text"):
                optimized.append(current_chunk)

        return optimized

    def _split_large_chunk(self, chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split a chunk that exceeds max_chunk_size.

        Args:
            chunk: Chunk to split

        Returns:
            List of split chunks
        """
        text = chunk.get("text", "")
        if len(text) <= self.max_chunk_size:
            return [chunk]

        # Try to split at paragraph boundaries first
        paragraphs = text.split("\n\n")
        if len(paragraphs) > 1:
            split_chunks = []
            current_text = ""
            chunk_id_base = chunk.get("chunk_id", "chunk_0")
            start_char = chunk.get("start_char", 0)

            for para in paragraphs:
                if len(current_text) + len(para) + 2 > self.max_chunk_size and current_text:
                    # Create chunk from accumulated text
                    split_chunk = chunk.copy()
                    split_chunk["text"] = current_text.strip()
                    split_chunk["chunk_id"] = f"{chunk_id_base}_part{len(split_chunks)}"
                    split_chunk["end_char"] = start_char + len(current_text)
                    split_chunks.append(split_chunk)
                    start_char += len(current_text)
                    current_text = para
                else:
                    current_text += "\n\n" + para if current_text else para

            # Add remaining text
            if current_text.strip():
                split_chunk = chunk.copy()
                split_chunk["text"] = current_text.strip()
                split_chunk["chunk_id"] = f"{chunk_id_base}_part{len(split_chunks)}"
                split_chunk["end_char"] = chunk.get("end_char", 0)
                split_chunks.append(split_chunk)

            return split_chunks

        # Fallback: split at sentence boundaries
        sentences = re.split(r'([.!?]\s+)', text)
        split_chunks = []
        current_text = ""
        chunk_id_base = chunk.get("chunk_id", "chunk_0")
        start_char = chunk.get("start_char", 0)

        for sentence in sentences:
            if len(current_text) + len(sentence) > self.max_chunk_size and current_text:
                split_chunk = chunk.copy()
                split_chunk["text"] = current_text.strip()
                split_chunk["chunk_id"] = f"{chunk_id_base}_part{len(split_chunks)}"
                split_chunk["end_char"] = start_char + len(current_text)
                split_chunks.append(split_chunk)
                start_char += len(current_text)
                current_text = sentence
            else:
                current_text += sentence

        if current_text.strip():
            split_chunk = chunk.copy()
            split_chunk["text"] = current_text.strip()
            split_chunk["chunk_id"] = f"{chunk_id_base}_part{len(split_chunks)}"
            split_chunk["end_char"] = chunk.get("end_char", 0)
            split_chunks.append(split_chunk)

        return split_chunks

    def _find_sentence_boundary(self, text: str, start: int, preferred_end: int) -> int:
        """
        Find a good sentence boundary near the preferred end position.

        Args:
            text: Full text
            start: Start position
            preferred_end: Preferred end position

        Returns:
            Actual end position at sentence boundary
        """
        window_size = min(200, preferred_end - start)
        search_start = max(start, preferred_end - window_size)

        # Find last sentence ending before preferred_end
        for i in range(preferred_end, search_start, -1):
            if i < len(text) and text[i] in ".!?":
                if i + 1 < len(text) and text[i + 1] in " \n":
                    return i + 1

        return preferred_end
