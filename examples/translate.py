#!/usr/bin/env python3
"""
TranslateGemma - Unified Translation Tool

Supports multiple backends and modes:
- Backends: transformers, ollama, mlx
- Modes: one-shot (single translation), interactive (REPL), pdf (PDF translation)

Usage:
    # One-shot translation (default backend: ollama)
    python translate.py --text "Hello, world!" --target zh-TW

    # Interactive mode
    python translate.py --mode interactive --backend transformers

    # PDF translation
    python translate.py --mode pdf --file examples/2601.09012v2.pdf --target zh-TW

    # Translate specific pages from PDF
    python translate.py --mode pdf --file doc.pdf --start-page 1 --end-page 3

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
from pathlib import Path

# Add backends to path
sys.path.insert(0, os.path.dirname(__file__))

from backends import get_backend, HAS_MLX

# Check if PyMuPDF is available
try:
    import fitz  # PyMuPDF
    HAS_PDF = True
except ImportError:
    HAS_PDF = False


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


def download_arxiv_pdf(arxiv_id: str, output_dir: str = "examples") -> str:
    """
    Download PDF from arXiv

    Args:
        arxiv_id: arXiv ID (e.g., "2601.09012v2" or "2601.09012")
        output_dir: Directory to save the PDF (default: "examples")

    Returns:
        Path to downloaded PDF file
    """
    import urllib.request
    import os

    # Clean arxiv_id (remove any "v" version suffix for URL)
    # arXiv URL format: https://arxiv.org/pdf/2601.09012.pdf
    base_id = arxiv_id.split('v')[0] if 'v' in arxiv_id else arxiv_id

    # Construct URL
    url = f"https://arxiv.org/pdf/{base_id}.pdf"

    # Create output directory if not exists
    os.makedirs(output_dir, exist_ok=True)

    # Output file path (keep version in filename)
    output_path = os.path.join(output_dir, f"{arxiv_id}.pdf")

    # Check if already downloaded
    if os.path.exists(output_path):
        print_info(f"PDF already exists: {output_path}")
        return output_path

    # Download
    print(f"Downloading from arXiv: {url}")
    try:
        urllib.request.urlretrieve(url, output_path)
        print_success(f"Downloaded: {output_path}")
        return output_path
    except Exception as e:
        print_error(f"Failed to download arXiv PDF: {arxiv_id}", str(e))
        raise


def extract_text_from_pdf(pdf_path: str, start_page: Optional[int] = None, end_page: Optional[int] = None) -> list[tuple[int, str]]:
    """
    Extract text from PDF file

    Args:
        pdf_path: Path to PDF file
        start_page: Starting page number (1-indexed, inclusive)
        end_page: Ending page number (1-indexed, inclusive)

    Returns:
        List of (page_number, text) tuples
    """
    if not HAS_PDF:
        raise ImportError("PyMuPDF not installed. Run: uv pip install pymupdf")

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    doc = fitz.open(pdf_path)
    total_pages = len(doc)

    # Determine page range
    if start_page is None:
        start_page = 1
    if end_page is None:
        end_page = total_pages

    # Validate page range
    if start_page < 1 or start_page > total_pages:
        raise ValueError(f"Invalid start page: {start_page} (PDF has {total_pages} pages)")
    if end_page < start_page or end_page > total_pages:
        raise ValueError(f"Invalid end page: {end_page} (must be >= {start_page} and <= {total_pages})")

    # Extract text from pages
    pages_text = []
    for page_num in range(start_page - 1, end_page):  # 0-indexed internally
        page = doc[page_num]
        text = page.get_text()
        pages_text.append((page_num + 1, text))  # Store 1-indexed page number

    doc.close()
    return pages_text


def pdf_pages_to_images(pdf_path: str, start_page: Optional[int] = None, end_page: Optional[int] = None, dpi: int = 150):
    """
    Convert PDF pages to images

    Args:
        pdf_path: Path to PDF file
        start_page: Starting page number (1-indexed, inclusive)
        end_page: Ending page number (1-indexed, inclusive)
        dpi: DPI for rendering (default: 150, higher = better quality but slower)

    Returns:
        List of (page_number, PIL_Image) tuples
    """
    if not HAS_PDF:
        raise ImportError("PyMuPDF not installed. Run: uv pip install pymupdf")

    try:
        from PIL import Image
    except ImportError:
        raise ImportError("Pillow not installed. Run: uv pip install pillow")

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    doc = fitz.open(pdf_path)
    total_pages = len(doc)

    # Determine page range
    if start_page is None:
        start_page = 1
    if end_page is None:
        end_page = total_pages

    # Validate page range
    if start_page < 1 or start_page > total_pages:
        raise ValueError(f"Invalid start page: {start_page} (PDF has {total_pages} pages)")
    if end_page < start_page or end_page > total_pages:
        raise ValueError(f"Invalid end page: {end_page} (must be >= {start_page} and <= {total_pages})")

    # Convert pages to images
    pages_images = []
    for page_num in range(start_page - 1, end_page):  # 0-indexed internally
        page = doc[page_num]
        # Render page to pixmap (image)
        pix = page.get_pixmap(dpi=dpi)
        # Convert to PIL Image
        img_data = pix.tobytes("ppm")
        img = Image.frombytes("RGB", [pix.width, pix.height], img_data)
        pages_images.append((page_num + 1, img))  # Store 1-indexed page number

    doc.close()
    return pages_images


def pdf_mode(backend_name: str, pdf_path: str, source: str, target: str,
             start_page: Optional[int] = None, end_page: Optional[int] = None,
             pdf_as_image: bool = False, dpi: int = 96):
    """PDF translation mode

    Args:
        backend_name: Backend to use (ollama, transformers, mlx)
        pdf_path: Path to PDF file
        source: Source language code
        target: Target language code
        start_page: Starting page number
        end_page: Ending page number
        pdf_as_image: If True, use image mode (multimodal TranslateGemma)
    """
    print(f"{Colors.BOLD}TranslateGemma - PDF Translation{Colors.NC}")

    if pdf_as_image:
        print(f"Mode: {Colors.YELLOW}Image (Multimodal) ⚠️  Experimental{Colors.NC}")
        print(f"Backend: {Colors.CYAN}transformers-multimodal{Colors.NC}")
    else:
        print(f"Mode: {Colors.CYAN}Text{Colors.NC}")
        print(f"Backend: {Colors.CYAN}{backend_name}{Colors.NC}")

    print(f"File: {Colors.CYAN}{pdf_path}{Colors.NC}")
    print()

    # Get backend (force multimodal for image mode)
    try:
        if pdf_as_image:
            from backends import TransformersMultimodalBackend
            backend = TransformersMultimodalBackend()
            print_info("Using multimodal backend for image translation")
        else:
            backend = get_backend(backend_name)
    except ValueError as e:
        print_error(str(e))
        return 1
    except ImportError as e:
        print_error("Failed to import multimodal backend", str(e))
        return 1

    # Extract content from PDF
    try:
        if pdf_as_image:
            print(f"Converting PDF pages to images (DPI: {dpi})...")
            pages_data = pdf_pages_to_images(pdf_path, start_page, end_page, dpi=dpi)
            print_success(f"Converted {len(pages_data)} page(s) to images")
        else:
            print("Extracting text from PDF...")
            pages_data = extract_text_from_pdf(pdf_path, start_page, end_page)
            print_success(f"Extracted text from {len(pages_data)} page(s)")
        print()
    except Exception as e:
        print_error("Failed to process PDF", str(e))
        return 1

    # Load model
    backend_display = "transformers-multimodal" if pdf_as_image else backend_name
    print(f"Loading {backend_display} backend...")
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
    if pdf_as_image:
        print(f"   Multimodal: {info.get('capabilities', 'text + image')}")
    print()

    # Translate each page
    total_time = 0
    total_tokens = 0
    translated_pages = 0

    for page_num, page_content in pages_data:
        print(f"{Colors.BOLD}Page {page_num}:{Colors.NC}")

        if pdf_as_image:
            # Image mode
            print(f"{Colors.CYAN}Translating image (size: {page_content.size})...{Colors.NC}")
            try:
                result = backend.translate_image(page_content, source, target)
            except AttributeError:
                print_error("Backend doesn't support image translation")
                return 1
        else:
            # Text mode
            # Skip empty pages
            if not page_content.strip():
                print(f"{Colors.YELLOW}Empty, skipped{Colors.NC}")
                continue

            print(f"{Colors.CYAN}Translating {len(page_content)} characters...{Colors.NC}")
            result = backend.translate(page_content, source, target)

        if "error" in result.get("metadata", {}):
            print_error(f"Translation failed for page {page_num}", result["metadata"]["error"])
            continue

        # Print translation
        print(f"{Colors.GREEN}{result['translation']}{Colors.NC}")
        print()
        mode_info = result['metadata'].get('mode', 'text')
        print(f"Time: {result['time']:.2f}s, Tokens: {result['tokens']}, Speed: {result['metadata'].get('tokens_per_second', 0):.1f} tok/s, Mode: {mode_info}")
        print()
        print("─" * 80)
        print()

        total_time += result['time']
        total_tokens += result['tokens']
        translated_pages += 1

    # Print summary
    print(f"{Colors.BOLD}Summary:{Colors.NC}")
    print(f"  Mode: {'Image (Multimodal)' if pdf_as_image else 'Text'}")
    print(f"  Pages translated: {translated_pages}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Total tokens: {total_tokens}")
    if total_time > 0:
        print(f"  Average speed: {total_tokens / total_time:.1f} tok/s")

    # Cleanup
    backend.cleanup()

    return 0


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
        choices=["one-shot", "interactive", "pdf"],
        help="Translation mode (default: one-shot)"
    )

    parser.add_argument(
        "--text",
        help="Text to translate (required for one-shot mode)"
    )

    parser.add_argument(
        "--file",
        help="PDF file to translate (required for pdf mode, mutually exclusive with --arxiv)"
    )

    parser.add_argument(
        "--arxiv",
        help="arXiv ID (e.g., 2601.09012v2 or 2601.09012) to download PDF automatically"
    )

    parser.add_argument(
        "--start-page",
        type=int,
        help="Starting page number for PDF (1-indexed, default: 1)"
    )

    parser.add_argument(
        "--end-page",
        type=int,
        help="Ending page number for PDF (1-indexed, default: last page)"
    )

    parser.add_argument(
        "--pdf-as-image",
        action="store_true",
        help="Use image mode for PDF (experimental, multimodal TranslateGemma)"
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=96,
        help="DPI for PDF to image conversion (default: 96, lower = faster, higher = better quality). TranslateGemma expects 896x896."
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

    if args.mode == "pdf":
        if not HAS_PDF:
            parser.error("PDF mode requires PyMuPDF. Run: uv pip install pymupdf")
        if not args.file and not args.arxiv:
            parser.error("--file or --arxiv is required for pdf mode")
        if args.file and args.arxiv:
            parser.error("--file and --arxiv are mutually exclusive")
        if args.pdf_as_image:
            try:
                from PIL import Image
            except ImportError:
                parser.error("Image mode requires Pillow. Run: uv pip install pillow")

    # Run appropriate mode
    if args.mode == "one-shot":
        return one_shot_mode(args.backend, args.text, args.source, args.target)
    elif args.mode == "pdf":
        # Download from arXiv if specified
        pdf_file = args.file
        if args.arxiv:
            pdf_file = download_arxiv_pdf(args.arxiv)

        return pdf_mode(args.backend, pdf_file, args.source, args.target,
                       args.start_page, args.end_page, args.pdf_as_image, args.dpi)
    else:
        return interactive_mode(args.backend)


if __name__ == "__main__":
    sys.exit(main())
