# OCR Input Parser Component

## Overview

The OCR Input Parser component transforms raw OCR output into a clean, structured, and semantically consistent format for downstream agents. This parser is critical for agent performance, as it directly impacts the accuracy and consistency of parsed output.

**Key Features:**
- Uses `layout_conserved_with_lineno` for all text content (with line numbers)
- Uses AWS Textract for table extraction (outputs HTML format)
- Removes OCR noise and inconsistencies
- Groups fields and key-value pairs
- Handles multi-page content

## Features

### 1. OCR Noise Removal and Data Cleaning
- Removes empty or invalid text entries
- Filters by confidence threshold
- Cleans common OCR artifacts and inconsistencies
- Validates and normalizes bounding boxes
- Handles missing or malformed data gracefully

### 2. Text Content Extraction
- **Uses `layout_conserved_with_lineno` method exclusively**
- Preserves layout while adding line numbers
- Format: `-><line_no><- <text content>`
- Maintains spatial relationships

### 3. Table Extraction (AWS Textract)
- Uses AWS Textract `analyze_document` with `FeatureTypes=["TABLES"]`
- Extracts table structure (rows, columns, cells)
- **Outputs tables in HTML format**
- Includes bounding box information
- Handles complex table structures

### 4. Field Grouping and Key-Value Extraction
- Detects key-value patterns (e.g., "Label: Value")
- Groups fields by section
- Extracts bounding boxes and confidence scores
- Handles multi-page field extraction

### 5. Multi-Page Content Handling
- Processes each page independently
- Merges fields and content across pages
- Maintains page-level metadata
- Preserves spatial relationships

### 6. Deterministic Output Schema
- Consistent structure across all documents
- Agent-friendly format with clear hierarchies
- Includes metadata for traceability

## Usage

### Basic Usage

```python
from app.services.parsers.ocr_input_parser import OCRInputParser

# Initialize parser
parser = OCRInputParser(
    min_confidence=0.5,
    enable_table_detection=True,
    enable_field_grouping=True,
    combine_words=True,
)

# Parse OCR data
result = parser.parse(
    raw_ocr=[df_page1, df_page2, ...],  # List of DataFrames, one per page
    page_scalars=[{width: 800, height: 600}, ...],  # Optional page metadata
    page_images=[image1, image2, ...]  # Required for table extraction
)

# Access results
content = result["content"]  # Full document text (layout_conserved_with_lineno)
tables = result["tables"]  # Detected tables (HTML format)
fields = result["fields"]  # Grouped fields
pages = result["pages"]  # Structured page data
```

### Integration with PDF Parser

The OCR input parser is automatically integrated into the PDF parser's `_parse` method:

```python
# In PDFParser._parse()
ocr_parser = OCRInputParser(...)
parsed_result = ocr_parser.parse(
    raw_ocr=raw_ocr, 
    page_scalars=page_scalars,
    page_images=page_images
)
```

## Output Schema

The parser returns a dictionary with the following structure:

```python
{
    "cleaned_ocr": List[pd.DataFrame],  # Cleaned OCR data per page
    "pages": [
        {
            "page_number": int,
            "content": str,  # Text with layout_conserved_with_lineno format
            "blocks": [...],
            "tables": [
                {
                    "table_id": str,
                    "page": int,
                    "html": str,  # HTML table format
                    "bbox": [x0, y0, x2, y2]
                }
            ],
            "fields": {...},
            "word_count": int,
            "bbox": [x0, y0, x2, y2]
        }
    ],
    "tables": [
        {
            "table_id": str,
            "page": int,
            "html": str,  # HTML table format
            "bbox": [x0, y0, x2, y2]
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
    "content": str,  # Full document text (layout_conserved_with_lineno)
    "metadata": {
        "total_pages": int,
        "total_words": int,
        "total_tables": int,
        "total_fields": int,
        "parsing_method": "layout_conserved_with_lineno"
    }
}
```

## Configuration Options

- `min_confidence`: Minimum confidence threshold (0.0-1.0, default: 0.5)
- `enable_table_detection`: Enable AWS table detection (default: True)
- `enable_field_grouping`: Enable field grouping (default: True)
- `combine_words`: Combine close words into phrases (default: True)
- `gap_factor`: Factor for word proximity when combining (0.0-1.0, default: 0.75)
- `aws_region`: AWS region for Textract (reads from `AWS_REGION` env var if None)
- `aws_access_key_id`: AWS access key ID (reads from `AWS_ACCESS_KEY_ID` env var if None)
- `aws_secret_access_key`: AWS secret access key (reads from `AWS_SECRET_ACCESS_KEY` env var if None)

## Text Format

All text content uses `layout_conserved_with_lineno` format:

```
->0<- Header text here
->1<- Some content on line 1
->2<- More content on line 2
```

Line numbers are zero-indexed and padded for alignment.

## Table Format

Tables are extracted using AWS Textract and output as HTML:

```html
<table>
<tr>
<td>Header 1</td>
<td>Header 2</td>
</tr>
<tr>
<td>Cell 1</td>
<td>Cell 2</td>
</tr>
</table>
```

## Dependencies

- `docllm`: Core parsing utilities and structures
- `boto3`: AWS SDK for Textract table extraction
- `pandas`: Data manipulation
- `loguru`: Logging

## AWS Setup

For table extraction to work, you need:

1. **AWS credentials** configured via environment variables:
   - `AWS_ACCESS_KEY_ID`: Your AWS access key ID
   - `AWS_SECRET_ACCESS_KEY`: Your AWS secret access key
   - `AWS_REGION`: AWS region (e.g., "us-east-1", "eu-west-1")

2. **AWS Textract access permissions** - Your AWS credentials must have permission to call `textract:AnalyzeDocument`

3. **Alternative**: You can also pass credentials directly to the parser:
   ```python
   parser = OCRInputParser(
       aws_region="us-east-1",
       aws_access_key_id="your-key-id",
       aws_secret_access_key="your-secret-key"
   )
   ```

The parser will automatically read from environment variables if credentials are not provided in the constructor.

## Error Handling

The parser includes comprehensive error handling:
- Gracefully handles empty or malformed input
- Falls back to simple extraction methods when advanced parsing fails
- Logs warnings for debugging while continuing processing
- Returns empty structures rather than raising exceptions
- AWS table extraction failures don't block text extraction

## Performance Considerations

- Processes pages independently (can be parallelized)
- Uses efficient pandas operations
- AWS Textract calls are made per page (consider rate limits)
- Minimizes data copying where possible
