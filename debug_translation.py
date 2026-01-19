#!/usr/bin/env python3
"""
Debug script to test translation extraction

Usage:
    python debug_translation.py
"""

import os
import sys

# Enable DEBUG mode
os.environ['TRANSLATE_DEBUG'] = '1'

# Add examples to path
sys.path.insert(0, 'examples')
sys.path.insert(0, 'examples/backends')

from transformers_backend import TransformersBackend
import fitz  # PyMuPDF


def extract_text_from_page(pdf_path: str, page_num: int) -> str:
    """Extract text from a specific page"""
    doc = fitz.open(pdf_path)
    page = doc[page_num - 1]  # 0-indexed
    text = page.get_text()
    doc.close()
    return text.strip()


def main():
    print("="*80)
    print("DEBUG: Translation Extraction Test")
    print("="*80)

    # Test with a specific PDF page
    pdf_path = os.path.expanduser("~/Desktop/2601.09012v2.pdf")
    page_num = 5  # Page 5 (Method section) - the problematic one

    if not os.path.exists(pdf_path):
        print(f"❌ PDF not found: {pdf_path}")
        print("Please check the PDF path")
        return

    # Extract text
    print(f"\n📄 Extracting text from Page {page_num}...")
    text = extract_text_from_page(pdf_path, page_num)
    print(f"✅ Extracted {len(text)} characters\n")
    print(f"Text preview:")
    print(text[:200])
    print("\n" + "="*80)

    # Load model
    print("\n🔄 Loading TranslateGemma model...")
    backend = TransformersBackend()
    backend.load_model()
    print("✅ Model loaded\n")

    # Translate
    print("="*80)
    print("🌐 Translating...")
    print("="*80)

    result = backend.translate(
        text,
        source_lang="en",
        target_lang="zh-TW"
    )

    print("\n" + "="*80)
    print("RESULT:")
    print("="*80)
    print(f"Translation ({len(result['translation'])} chars):")
    print(result['translation'])
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
