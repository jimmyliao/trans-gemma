#!/usr/bin/env python3
"""
TranslateGemma - Unified Translation Tool

Supports multiple backends and modes:
- Backends: transformers, ollama, mlx
- Modes: one-shot (single translation), interactive (REPL)

Usage:
    # One-shot translation (default backend: ollama)
    python translate.py --text "Hello, world!" --target zh-TW

    # Interactive mode
    python translate.py --mode interactive --backend transformers

    # Specify backend
    python translate.py --backend mlx --text "How are you?"

Environment Variables:
    BACKEND: Default backend (transformers, ollama, mlx)
    FORCE_DEVICE: Device for transformers (cpu, mps, auto)
    NO_MEM_LIMIT: Disable memory limit for transformers (0, 1)
"""

import argparse
import os
import sys
import time
from typing import Optional

# Add backends to path
sys.path.insert(0, os.path.dirname(__file__))

from backends import get_backend, HAS_MLX


# Colors for terminal output
class Colors:
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    CYAN = '\033[0;36m'
    RED = '\033[0;31m'
    BOLD = '\033[1m'
    NC = '\033[0m'  # No Color


def print_success(message, detail=""):
    print(f"{Colors.GREEN}✅ {message}{Colors.NC}")
    if detail:
        print(f"   {detail}")


def print_error(message, detail=""):
    print(f"{Colors.RED}❌ {message}{Colors.NC}")
    if detail:
        print(f"   {detail}")


def print_warning(message):
    print(f"{Colors.YELLOW}⚠️  {message}{Colors.NC}")


def print_info(message):
    print(f"{Colors.BLUE}ℹ️  {message}{Colors.NC}")


def one_shot_mode(backend_name: str, text: str, source: str, target: str):
    """One-shot translation mode"""
    print(f"{Colors.BOLD}TranslateGemma - One-shot Translation{Colors.NC}")
    print(f"Backend: {Colors.CYAN}{backend_name}{Colors.NC}")
    print()

    # Get backend
    try:
        backend = get_backend(backend_name)
    except ValueError as e:
        print_error(str(e))
        return 1

    # Load model
    print(f"Loading {backend_name} backend...")
    result = backend.load_model()

    if not result["model_loaded"]:
        error = result.get("metadata", {}).get("error", "Unknown error")
        print_error("Failed to load model", error)
        return 1

    print_success(
        "Model loaded",
        f"Time: {result['load_time']:.2f}s"
    )

    # Get backend info
    info = backend.get_backend_info()
    print(f"   Device: {info.get('device', 'unknown')}")
    print()

    # Translate
    print(f"Translating: {Colors.CYAN}{text}{Colors.NC}")
    print(f"Target: {target}")
    print()

    result = backend.translate(text, source, target)

    if "error" in result.get("metadata", {}):
        print_error("Translation failed", result["metadata"]["error"])
        return 1

    # Print results
    print(f"{Colors.BOLD}Translation:{Colors.NC}")
    print(f"  {Colors.GREEN}{result['translation']}{Colors.NC}")
    print()
    print(f"Time: {result['time']:.2f}s")
    print(f"Tokens: {result['tokens']}")
    print(f"Speed: {result['metadata'].get('tokens_per_second', 0):.1f} tokens/s")

    # Cleanup
    backend.cleanup()

    return 0


def interactive_mode(backend_name: str):
    """Interactive REPL mode"""
    print(f"{Colors.BOLD}TranslateGemma - Interactive Mode{Colors.NC}")
    print(f"Backend: {Colors.CYAN}{backend_name}{Colors.NC}")
    print()

    # Get backend
    try:
        backend = get_backend(backend_name)
    except ValueError as e:
        print_error(str(e))
        return 1

    # Load model
    print(f"Loading {backend_name} backend...")
    result = backend.load_model()

    if not result["model_loaded"]:
        error = result.get("metadata", {}).get("error", "Unknown error")
        print_error("Failed to load model", error)
        return 1

    print_success(
        "Model loaded",
        f"Time: {result['load_time']:.2f}s"
    )

    # Get backend info
    info = backend.get_backend_info()
    print(f"   Device: {info.get('device', 'unknown')}")
    print()

    # Interactive loop
    print(f"{Colors.CYAN}Interactive mode - Type your text to translate{Colors.NC}")
    print(f"{Colors.YELLOW}Commands:{Colors.NC}")
    print(f"  :target <code>  - Change target language (default: zh-TW)")
    print(f"  :source <code>  - Change source language (default: en)")
    print(f"  :info           - Show backend info")
    print(f"  :quit, :exit    - Exit")
    print()

    source_lang = "en"
    target_lang = "zh-TW"
    total_translations = 0
    total_time = 0

    while True:
        try:
            # Get input
            print(f"{Colors.BOLD}[{source_lang} → {target_lang}]{Colors.NC} ", end="")
            user_input = input().strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.startswith(':'):
                cmd_parts = user_input[1:].split(maxsplit=1)
                cmd = cmd_parts[0].lower()

                if cmd in ['quit', 'exit', 'q']:
                    print()
                    print(f"{Colors.CYAN}Statistics:{Colors.NC}")
                    print(f"  Total translations: {total_translations}")
                    if total_translations > 0:
                        print(f"  Average time: {total_time / total_translations:.2f}s")
                    print()
                    print_success("Goodbye!")
                    break

                elif cmd == 'target':
                    if len(cmd_parts) > 1:
                        target_lang = cmd_parts[1]
                        print_info(f"Target language changed to: {target_lang}")
                    else:
                        print_warning("Usage: :target <language_code>")

                elif cmd == 'source':
                    if len(cmd_parts) > 1:
                        source_lang = cmd_parts[1]
                        print_info(f"Source language changed to: {source_lang}")
                    else:
                        print_warning("Usage: :source <language_code>")

                elif cmd == 'info':
                    print()
                    print(f"{Colors.CYAN}Backend Information:{Colors.NC}")
                    for key, value in info.items():
                        print(f"  {key}: {value}")
                    print()

                else:
                    print_warning(f"Unknown command: {cmd}")

                continue

            # Translate
            start_time = time.time()
            result = backend.translate(user_input, source_lang, target_lang)
            end_time = time.time()

            if "error" in result.get("metadata", {}):
                print_error("Translation failed", result["metadata"]["error"])
                continue

            # Print translation
            print(f"{Colors.GREEN}→ {result['translation']}{Colors.NC}")
            print(f"  ({result['time']:.2f}s, {result['metadata'].get('tokens_per_second', 0):.1f} tok/s)")
            print()

            total_translations += 1
            total_time += result['time']

        except KeyboardInterrupt:
            print()
            print()
            print_info("Interrupted. Type :quit to exit.")
            print()
        except EOFError:
            print()
            break

    # Cleanup
    backend.cleanup()

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="TranslateGemma - Multi-backend Translation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--backend",
        default=os.getenv("BACKEND", "ollama"),
        choices=["transformers", "ollama", "mlx"] if HAS_MLX else ["transformers", "ollama"],
        help="Translation backend (default: ollama)"
    )

    parser.add_argument(
        "--mode",
        default="one-shot",
        choices=["one-shot", "interactive"],
        help="Translation mode (default: one-shot)"
    )

    parser.add_argument(
        "--text",
        help="Text to translate (required for one-shot mode)"
    )

    parser.add_argument(
        "--source",
        default="en",
        help="Source language code (default: en)"
    )

    parser.add_argument(
        "--target",
        default="zh-TW",
        help="Target language code (default: zh-TW)"
    )

    args = parser.parse_args()

    # Validate arguments
    if args.mode == "one-shot" and not args.text:
        parser.error("--text is required for one-shot mode")

    # Run appropriate mode
    if args.mode == "one-shot":
        return one_shot_mode(args.backend, args.text, args.source, args.target)
    else:
        return interactive_mode(args.backend)


if __name__ == "__main__":
    sys.exit(main())
