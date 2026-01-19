#!/usr/bin/env python3
"""
Test full PDF translation (7 pages, same as Colab)

Usage:
    python test_full_pdf.py --backend ollama
    python test_full_pdf.py --backend transformers
    python test_full_pdf.py --backend both
"""

import os
import sys
import argparse
import time
from datetime import datetime

# Enable DEBUG
os.environ['TRANSLATE_DEBUG'] = '1'

sys.path.insert(0, 'examples')
sys.path.insert(0, 'examples/backends')

def test_with_backend(backend_name, pdf_path):
    """Test translation with specified backend"""
    print("\n" + "="*80)
    print(f"Testing {backend_name.upper()} Backend - Full PDF Translation")
    print("="*80)
    
    if backend_name == "ollama":
        from ollama_backend import OllamaBackend
        backend = OllamaBackend()
    else:
        os.environ['FORCE_DEVICE'] = 'mps'
        from transformers_backend import TransformersBackend
        backend = TransformersBackend()
    
    # Load model
    print(f"\n🔄 Loading model...")
    start = time.time()
    info = backend.load_model()
    load_time = time.time() - start
    print(f"✅ Model loaded in {load_time:.1f}s")
    print(f"   Metadata: {info.get('metadata', {})}")
    
    # Test sections (same as Colab)
    sections = {
        "abstract": (1, 1),
        "method": (3, 5),
        "experiments": (7, 9),
    }
    
    # Extract and translate
    import fitz
    
    results = []
    total_time = 0
    total_tokens = 0
    
    for section_name, (start_page, end_page) in sections.items():
        print(f"\n📖 Translating {section_name.upper()} (Pages {start_page}-{end_page})...")
        
        for page_num in range(start_page, end_page + 1):
            # Extract text
            doc = fitz.open(pdf_path)
            page = doc[page_num - 1]
            text = page.get_text().strip()
            doc.close()
            
            if not text:
                print(f"   ⚠️  Page {page_num}: No text")
                continue
            
            print(f"   📄 Page {page_num}: {len(text)} chars extracted")
            
            # Translate
            result = backend.translate(text, "en", "zh-TW")
            
            total_time += result['time']
            total_tokens += result.get('tokens', 0)
            
            # Check if translation is valid
            translation = result['translation']
            is_valid = len(translation) > 50 and any('\u4e00' <= c <= '\u9fff' for c in translation)
            
            status = "✅" if is_valid else "❌"
            print(f"   {status} Page {page_num}: {result['time']:.1f}s, {len(translation)} chars")
            print(f"      Preview: {translation[:100]}...")
            
            results.append({
                'page': page_num,
                'section': section_name,
                'time': result['time'],
                'tokens': result.get('tokens', 0),
                'translation_length': len(translation),
                'valid': is_valid
            })
    
    # Summary
    print("\n" + "="*80)
    print(f"SUMMARY - {backend_name.upper()}")
    print("="*80)
    print(f"Total pages translated: {len(results)}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"Average time/page: {total_time/len(results):.1f}s")
    print(f"Total tokens: {total_tokens}")
    print(f"Tokens/second: {total_tokens/total_time:.1f}")
    
    # Validation
    valid_count = sum(1 for r in results if r['valid'])
    print(f"\n✅ Valid translations: {valid_count}/{len(results)}")
    if valid_count < len(results):
        print("❌ Failed pages:")
        for r in results:
            if not r['valid']:
                print(f"   Page {r['page']}: {r['translation_length']} chars")
    
    return results, total_time

def main():
    parser = argparse.ArgumentParser(description="Test full PDF translation")
    parser.add_argument(
        "--backend",
        choices=["ollama", "transformers", "both"],
        default="ollama",
        help="Which backend to test"
    )
    parser.add_argument(
        "--pdf",
        default=os.path.expanduser("~/Desktop/2601.09012v2.pdf"),
        help="Path to PDF file"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.pdf):
        print(f"❌ PDF not found: {args.pdf}")
        return
    
    print(f"\n📄 Testing with PDF: {args.pdf}")
    print(f"🕐 Started at: {datetime.now().strftime('%H:%M:%S')}")
    
    results = {}
    
    if args.backend in ["ollama", "both"]:
        try:
            results['ollama'] = test_with_backend("ollama", args.pdf)
        except Exception as e:
            print(f"\n❌ Ollama test failed: {e}")
            import traceback
            traceback.print_exc()
    
    if args.backend in ["transformers", "both"]:
        try:
            results['transformers'] = test_with_backend("transformers", args.pdf)
        except Exception as e:
            print(f"\n❌ Transformers test failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Comparison
    if len(results) == 2:
        print("\n" + "="*80)
        print("COMPARISON")
        print("="*80)
        
        ollama_results, ollama_time = results['ollama']
        tf_results, tf_time = results['transformers']
        
        print(f"\nOllama:")
        print(f"  Total time: {ollama_time:.1f}s")
        print(f"  Avg/page: {ollama_time/len(ollama_results):.1f}s")
        
        print(f"\nHuggingFace:")
        print(f"  Total time: {tf_time:.1f}s")
        print(f"  Avg/page: {tf_time/len(tf_results):.1f}s")
        
        speedup = tf_time / ollama_time
        print(f"\n⚡ Ollama is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'}")
    
    print(f"\n🕐 Completed at: {datetime.now().strftime('%H:%M:%S')}")

if __name__ == "__main__":
    main()
