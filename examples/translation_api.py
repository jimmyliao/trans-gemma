"""
Simple API wrapper for using TranslateGemma in notebooks (Colab, Jupyter, etc.)

Usage in Colab/Jupyter:
    from translation_api import TranslateGemma

    # Initialize with default backend (auto-detects environment)
    translator = TranslateGemma()

    # Translate text
    result = translator.translate("Hello, world!", target="zh-TW")
    print(result.translation)  # 你好，世界！

    # Specify backend explicitly
    translator = TranslateGemma(backend="transformers")
    result = translator.translate("Good morning", target="ja")
    print(result.translation)
"""

import os
import sys
from typing import Optional, Dict, Any
from dataclasses import dataclass


# Add backends to path
sys.path.insert(0, os.path.dirname(__file__))

from backends import get_backend, HAS_MLX


@dataclass
class TranslationResult:
    """Translation result with metadata"""
    translation: str
    time: float
    tokens: int
    tokens_per_second: float
    backend: str
    metadata: Dict[str, Any]


class TranslateGemma:
    """
    Simple API for TranslateGemma translation in notebooks

    Examples:
        # Auto-detect backend
        translator = TranslateGemma()

        # Specify backend
        translator = TranslateGemma(backend="transformers")
        translator = TranslateGemma(backend="ollama")  # Fast on M1
        translator = TranslateGemma(backend="mlx")     # Fastest on M1

        # Translate
        result = translator.translate("Hello!", target="zh-TW")
        print(result.translation)
        print(f"Time: {result.time:.2f}s, Speed: {result.tokens_per_second:.1f} tok/s")
    """

    def __init__(self, backend: Optional[str] = None):
        """
        Initialize TranslateGemma translator

        Args:
            backend: Backend to use. If None, auto-detects:
                - Colab/Linux CUDA: transformers
                - M1/M2/M3 Mac: ollama (if installed) or mlx
                - Other: transformers
        """
        if backend is None:
            backend = self._auto_detect_backend()

        self.backend_name = backend
        self.backend = get_backend(backend)
        self._loaded = False

    def _auto_detect_backend(self) -> str:
        """Auto-detect best backend for current environment"""
        import platform

        # Check if running in Colab
        try:
            import google.colab
            return "transformers"  # Colab has T4 GPU
        except:
            pass

        # Check platform
        system = platform.system()

        if system == "Darwin":  # macOS
            # Check if M1/M2/M3
            machine = platform.machine()
            if machine == "arm64":
                # Check if Ollama is installed
                import subprocess
                try:
                    result = subprocess.run(
                        ["ollama", "list"],
                        capture_output=True,
                        timeout=2
                    )
                    if result.returncode == 0:
                        return "ollama"  # Ollama available
                except:
                    pass

                # Check if MLX is installed
                if HAS_MLX:
                    return "mlx"

        # Default to transformers
        return "transformers"

    def _ensure_loaded(self):
        """Ensure model is loaded"""
        if not self._loaded:
            print(f"Loading {self.backend_name} backend...")
            result = self.backend.load_model()

            if not result["model_loaded"]:
                error = result.get("metadata", {}).get("error", "Unknown error")
                raise RuntimeError(f"Failed to load model: {error}")

            print(f"✅ Model loaded in {result['load_time']:.2f}s")
            info = self.backend.get_backend_info()
            print(f"   Device: {info.get('device', 'unknown')}")
            self._loaded = True

    def translate(
        self,
        text: str,
        source: str = "en",
        target: str = "zh-TW"
    ) -> TranslationResult:
        """
        Translate text

        Args:
            text: Text to translate
            source: Source language code (ISO 639-1), default: "en"
            target: Target language code (ISO 639-1), default: "zh-TW"

        Returns:
            TranslationResult with translation and metadata

        Examples:
            result = translator.translate("Hello!", target="ja")
            result = translator.translate("Bonjour", source="fr", target="en")
        """
        self._ensure_loaded()

        result = self.backend.translate(text, source, target)

        if "error" in result.get("metadata", {}):
            raise RuntimeError(f"Translation failed: {result['metadata']['error']}")

        return TranslationResult(
            translation=result["translation"],
            time=result["time"],
            tokens=result["tokens"],
            tokens_per_second=result["metadata"].get("tokens_per_second", 0),
            backend=self.backend_name,
            metadata=result["metadata"]
        )

    def batch_translate(
        self,
        texts: list[str],
        source: str = "en",
        target: str = "zh-TW"
    ) -> list[TranslationResult]:
        """
        Translate multiple texts

        Args:
            texts: List of texts to translate
            source: Source language code
            target: Target language code

        Returns:
            List of TranslationResult

        Example:
            texts = ["Hello", "Good morning", "Thank you"]
            results = translator.batch_translate(texts, target="ja")
            for result in results:
                print(result.translation)
        """
        return [self.translate(text, source, target) for text in texts]

    def get_info(self) -> Dict[str, str]:
        """Get backend information"""
        return self.backend.get_backend_info()

    def cleanup(self):
        """Clean up model from memory"""
        self.backend.cleanup()
        self._loaded = False

    def __enter__(self):
        """Context manager support"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up on exit"""
        self.cleanup()


# Convenience function for quick translations
def translate(text: str, target: str = "zh-TW", backend: Optional[str] = None) -> str:
    """
    Quick translation function

    Args:
        text: Text to translate
        target: Target language code
        backend: Backend to use (auto-detect if None)

    Returns:
        Translated text

    Example:
        translation = translate("Hello, world!", target="ja")
        print(translation)  # こんにちは、世界！
    """
    with TranslateGemma(backend=backend) as translator:
        result = translator.translate(text, target=target)
        return result.translation


if __name__ == "__main__":
    # Example usage
    print("TranslateGemma API Test\n")

    # Test auto-detect
    translator = TranslateGemma()
    print(f"Auto-detected backend: {translator.backend_name}\n")

    # Test translation
    texts = [
        "Hello, world!",
        "Good morning",
        "How are you?"
    ]

    print("Translations to Traditional Chinese (zh-TW):")
    for text in texts:
        result = translator.translate(text, target="zh-TW")
        print(f"  {text} → {result.translation} ({result.time:.2f}s)")

    print("\nTranslations to Japanese (ja):")
    for text in texts:
        result = translator.translate(text, target="ja")
        print(f"  {text} → {result.translation} ({result.time:.2f}s)")

    translator.cleanup()
