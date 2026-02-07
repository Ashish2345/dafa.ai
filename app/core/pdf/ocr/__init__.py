"""
OCR module for text extraction from PDF documents.

This module provides OCR implementations for different types of PDF documents:

- **DigitalOCR**: For digitally-created PDFs with embedded text
- **GoogleVisionOCR**: Synchronous Google Cloud Vision OCR for scanned documents
- **GoogleVisionOCRAsync**: Asynchronous Google Cloud Vision OCR for scanned documents
- **AWSTextractOCR**: Synchronous AWS Textract OCR for scanned documents
- **AWSTextractOCRAsync**: Asynchronous AWS Textract OCR for scanned documents

Example usage:
    Digital PDF:
        from diu_new.pdf_reader import PDFReader
        from diu_new.ocr import DigitalOCR

        ocr = DigitalOCR(reset_lines_and_sort=True)
        with PDFReader("document.pdf") as reader:
            df_list = ocr.process_document(reader)

    Scanned PDF (Google Vision - Sync):
        from diu_new.ocr import GoogleVisionOCR

        ocr = GoogleVisionOCR(jpeg_quality=85, fix_orientation=True)
        with PDFReader("scanned.pdf") as reader:
            df_list = ocr.process_document(reader)

    Scanned PDF (Google Vision - Async):
        from diu_new.ocr import GoogleVisionOCRAsync

        async_ocr = GoogleVisionOCRAsync()
        df = await async_ocr.process_image_async(image)

    Scanned PDF (AWS Textract - Sync):
        from diu_new.ocr import AWSTextractOCR

        ocr = AWSTextractOCR(jpeg_quality=85, fix_orientation=True)
        with PDFReader("scanned.pdf") as reader:
            df_list = ocr.process_document(reader)

    Scanned PDF (AWS Textract - Async):
        from diu_new.ocr import AWSTextractOCRAsync

        async_ocr = AWSTextractOCRAsync()
        df = await async_ocr.process_image_async(image)
"""

from .base import OCR_COLUMNS, OCRBase
from .digital import DigitalOCR
from .non_digital import (
    AWSTextractOCR,
    AWSTextractOCRAsync,
    GoogleVisionOCR,
    GoogleVisionOCRAsync,
)

__all__ = [
    "OCRBase",
    "OCR_COLUMNS",
    "DigitalOCR",
    "GoogleVisionOCR",
    "GoogleVisionOCRAsync",
    "AWSTextractOCR",
    "AWSTextractOCRAsync",
]
