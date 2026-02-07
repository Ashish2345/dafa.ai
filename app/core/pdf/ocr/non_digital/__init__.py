"""
Non-digital OCR module for scanned documents and images.

This module contains OCR implementations for scanned documents and images
that require optical character recognition to extract text.

Available implementations:
- GoogleVisionOCR: Synchronous Google Cloud Vision API OCR
- GoogleVisionOCRAsync: Asynchronous Google Cloud Vision API OCR
- AWSTextractOCR: Synchronous AWS Textract API OCR
- AWSTextractOCRAsync: Asynchronous AWS Textract API OCR
- AzureComputerVisionOCR: Synchronous Azure Computer Vision API OCR
- AzureComputerVisionOCRAsync: Asynchronous Azure Computer Vision API OCR

Example usage:
    Synchronous:
        from diu_new.ocr.non_digital import GoogleVisionOCR, AWSTextractOCR, AzureComputerVisionOCR
        from diu_new.pdf_reader import PDFReader

        # Google Vision
        ocr = GoogleVisionOCR(jpeg_quality=85, fix_orientation=True)
        with PDFReader("scanned_document.pdf") as reader:
            df_list = ocr.process_document(reader)

        # AWS Textract
        ocr = AWSTextractOCR(jpeg_quality=85, fix_orientation=True)
        with PDFReader("scanned_document.pdf") as reader:
            df_list = ocr.process_document(reader)

        # Azure Computer Vision
        ocr = AzureComputerVisionOCR(jpeg_quality=85, fix_orientation=True)
        with PDFReader("scanned_document.pdf") as reader:
            df_list = ocr.process_document(reader)

    Asynchronous:
        from diu_new.ocr.non_digital import GoogleVisionOCRAsync, AWSTextractOCRAsync, AzureComputerVisionOCRAsync

        # Google Vision
        async_ocr = GoogleVisionOCRAsync()
        df = await async_ocr.process_image_async(image)

        # AWS Textract
        async_ocr = AWSTextractOCRAsync()
        df = await async_ocr.process_image_async(image)

        # Azure Computer Vision
        async_ocr = AzureComputerVisionOCRAsync()
        df = await async_ocr.process_image_async(image)
"""

from .aws import AWSTextractOCR, AWSTextractOCRAsync
from .google import GoogleVisionOCR, GoogleVisionOCRAsync
from .azure import AzureComputerVisionOCR, AzureComputerVisionOCRAsync

__all__ = [
    "GoogleVisionOCR",
    "GoogleVisionOCRAsync",
    "AWSTextractOCR",
    "AWSTextractOCRAsync",
    "AzureComputerVisionOCR",
    "AzureComputerVisionOCRAsync",
]
