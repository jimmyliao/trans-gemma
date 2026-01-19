#!/usr/bin/env python3
"""
Test Ollama vs HuggingFace backends for TranslateGemma

Usage:
    # Test Ollama
    python test_ollama_vs_hf.py --backend ollama
    
    # Test HuggingFace (fixed MPS)
    FORCE_DEVICE=mps python test_ollama_vs_hf.py --backend transformers
"""

import sys
import argparse
import time

sys.path.insert(0, 'examples')
sys.path.insert(0, 'examples/backends')


def test_ollama():
    """Test Ollama backend"""
    print("="*80)
    print("Testing Ollama Backend")
    print("="*80)
    
    from ollama_backend import OllamaBackend
    
    backend = OllamaBackend()
    
    # Load model
    print("\n🔄 Checking Ollama...")
    try:
        info = backend.load_model()
        print(f"✅ Ollama ready: {info['metadata']}")
    except Exception as e:
        print(f"❌ Ollama error: {e}")
        print("\n💡 Setup instructions:")
        print("   1. brew install ollama")
        print("   2. ollama serve")
        print("   3. ollama pull translategemma")
        return None
    
    # Test translation
    test_text = "Hello, how are you today?"
    print(f"\n📝 Test text: {test_text}")
    print("🌐 Translating en → zh-TW...")
    
    result = backend.translate(test_text, "en", "zh-TW")
    
    print(f"\n✅ Translation: {result['translation']}")
    print(f"⏱️  Time: {result['time']:.2f}s")
    print(f"🔢 Tokens: {result['tokens']}")
    print(f"⚡ Speed: {result['metadata']['tokens_per_second']:.1f} tokens/s")
    
    return result


def test_transformers():
    """Test HuggingFace Transformers backend (with MPS fix)"""
    print("="*80)
    print("Testing HuggingFace Transformers Backend")
    print("="*80)
    
    import os
    os.environ['FORCE_DEVICE'] = 'mps'  # Force MPS for M1
    
    from transformers_backend import TransformersBackend
    
    backend = TransformersBackend()
    
    # Load model
    print("\n🔄 Loading TranslateGemma (HuggingFace)...")
    try:
        info = backend.load_model()
        print(f"✅ Model loaded: {info['metadata']}")
    except Exception as e:
        print(f"❌ Error: {e}")
        return None
    
    # Test translation
    test_text = "Hello, how are you today?"
    print(f"\n📝 Test text: {test_text}")
    print("🌐 Translating en → zh-TW...")
    
    result = backend.translate(test_text, "en", "zh-TW")
    
    print(f"\n✅ Translation: {result['translation']}")
    print(f"⏱️  Time: {result['time']:.2f}s")
    print(f"🔢 Tokens: {result['tokens']}")
    print(f"⚡ Speed: {result['metadata']['tokens_per_second']:.1f} tokens/s")
    
    backend.cleanup()
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Test TranslateGemma backends")
    parser.add_argument(
        "--backend",
        choices=["ollama", "transformers", "both"],
        default="both",
        help="Which backend to test"
    )
    
    args = parser.parse_args()
    
    results = {}
    
    if args.backend in ["ollama", "both"]:
        results['ollama'] = test_ollama()
        print("\n")
    
    if args.backend in ["transformers", "both"]:
        results['transformers'] = test_transformers()
        print("\n")
    
    # Comparison
    if len(results) == 2 and all(results.values()):
        print("="*80)
        print("COMPARISON")
        print("="*80)
        print(f"\nOllama:")
        print(f"  Time: {results['ollama']['time']:.2f}s")
        print(f"  Translation: {results['ollama']['translation']}")
        
        print(f"\nHuggingFace:")
        print(f"  Time: {results['transformers']['time']:.2f}s")
        print(f"  Translation: {results['transformers']['translation']}")
        
        speedup = results['transformers']['time'] / results['ollama']['time']
        print(f"\n⚡ Ollama is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'}")


if __name__ == "__main__":
    main()
