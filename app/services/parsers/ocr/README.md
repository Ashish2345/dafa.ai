# OCR Parser Component

## Overview

The OCR Parser component transforms raw OCR output into a clean, structured, and semantically consistent format for downstream agents. This parser is critical for agent performance, as it directly impacts the accuracy and consistency of parsed output.

## Features

### 1. OCR Noise Removal and Data Cleaning
- Removes empty or invalid text entries
- Filters by confidence threshold
- Cleans common OCR artifacts and inconsistencies
- Validates and normalizes bounding boxes
- Handles missing or malformed data gracefully

### 2. Field Grouping and Key-Value Extraction
- Detects key-value patterns (e.g., "Label: Value")
- Groups fields by section
- Extracts bounding boxes and confidence scores
- Handles multi-page field extraction

### 3. Table Detection
- Identifies table-like structures in OCR data
- Detects column alignment patterns
- Estimates table dimensions (rows, columns)
- Calculates table bounding boxes
- Groups table content by blocks

### 4. Multi-Page Content Handling
- Processes each page independently
- Merges fields and content across pages
- Maintains page-level metadata
- Preserves spatial relationships

### 5. Deterministic Output Schema
- Consistent structure across all documents
- Agent-friendly format with clear hierarchies
- Includes metadata for traceability
- Supports both structured and fallback modes

## Usage

### Basic Usage

```python
from app.services.parsers.ocr import OCRParser

# Initialize parser
parser = OCRParser(
    input_parsing_method="layout_conserved_advance",
    min_confidence=0.5,
    enable_table_detection=True,
    enable_field_grouping=True,
)

# Parse OCR data
result = parser.parse(
    raw_ocr=[df_page1, df_page2, ...],  # List of DataFrames, one per page
    page_scalars=[{width: 800, height: 600}, ...]  # Optional page metadata
)

# Access results
content = result["content"]  # Full document text
tables = result["tables"]  # Detected tables
fields = result["fields"]  # Grouped fields
pages = result["pages"]  # Structured page data
```

### Integration with PDF Parser

The OCR parser is automatically integrated into the PDF parser's `_parse` method:

```python
# In PDFParser._parse()
ocr_parser = OCRParser(...)
parsed_result = ocr_parser.parse(raw_ocr=raw_ocr, page_scalars=page_scalars)
```

## Output Schema

The parser returns a dictionary with the following structure:

```python
{
    "cleaned_ocr": List[pd.DataFrame],  # Cleaned OCR data per page
    "pages": [
        {
            "page_number": int,
            "content": str,
            "blocks": [
                {
                    "block_id": str,
                    "content": str,
                    "bbox": [x0, y0, x2, y2],
                    "word_count": int
                }
            ],
            "tables": [...],
            "fields": {...},
            "word_count": int,
            "bbox": [x0, y0, x2, y2]
        }
    ],
    "tables": [
        {
            "table_id": str,
            "page": int,
            "block": int,
            "bbox": [x0, y0, x2, y2],
            "row_count": int,
            "column_count": int
        }
    ],
    "fields": {
        "section_name": {
            "field_name": {
                "value": str,
                "key": str,
                "bbox": [x0, y0, x2, y2],
                "page": int,
                "confidence": float
            }
        }
    },
    "content": str,  # Full document text
    "metadata": {
        "total_pages": int,
        "total_words": int,
        "total_tables": int,
        "total_fields": int,
        "parsing_method": str
    }
}
```

## Configuration Options

- `input_parsing_method`: Method for parsing OCR input (default: "layout_conserved_advance")
  - Options: "layout_conserved_advance", "layout_conserved", "line_based", etc.
- `min_confidence`: Minimum confidence threshold (0.0-1.0, default: 0.5)
- `enable_table_detection`: Enable table detection (default: True)
- `enable_field_grouping`: Enable field grouping (default: True)
- `combine_words`: Combine close words into phrases (default: True)
- `gap_factor`: Factor for word proximity when combining (0.0-1.0, default: 0.75)

## Dependencies

- `docllm`: Core parsing utilities and structures
- `pandas`: Data manipulation
- `loguru`: Logging

## Error Handling

The parser includes comprehensive error handling:
- Gracefully handles empty or malformed input
- Falls back to simple extraction methods when advanced parsing fails
- Logs warnings for debugging while continuing processing
- Returns empty structures rather than raising exceptions

## Performance Considerations

- Processes pages independently (can be parallelized)
- Uses efficient pandas operations
- Caches parsing method functions
- Minimizes data copying where possible

## Future Enhancements

Potential improvements:
- Advanced table structure detection using ML
- Better field extraction with NLP
- Support for more OCR output formats
- Configurable cleaning rules
- Performance optimizations for large documents
