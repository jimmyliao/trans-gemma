"""MLX backend for TranslateGemma (Apple Silicon optimized)"""
import time
from typing import Dict, Any
from .base import TranslationBackend


class MLXBackend(TranslationBackend):
    """MLX backend for Apple Silicon

    Note: TranslateGemma MLX version not yet available.
    This backend is currently disabled.
    Use Ollama backend for best M1/M2/M3 performance.
    """

    def __init__(self):
        super().__init__()
        # MLX-optimized version not available yet
        # Using standard model (will fail for now)
        self.model_id = "google/translategemma-4b-it"

    def load_model(self, **kwargs) -> Dict[str, Any]:
        """Load model using MLX"""
        # TranslateGemma MLX version not available yet
        return {
            "model_loaded": False,
            "load_time": 0,
            "metadata": {
                "error": "TranslateGemma MLX version not yet available on Hugging Face.\n"
                        "   Use --backend ollama for best M1/M2/M3 performance."
            }
        }

    def translate(
        self,
        text: str,
        source_lang: str = "en",
        target_lang: str = "zh-TW"
    ) -> Dict[str, Any]:
        """Translate using MLX"""
        from mlx_lm import generate

        # Build prompt (similar to transformers)
        lang_names = {
            "en": "English",
            "zh-TW": "Traditional Chinese (Taiwan)",
            "zh-CN": "Simplified Chinese",
            "ja": "Japanese"
        }
        target_name = lang_names.get(target_lang, target_lang)
        prompt = f"Translate to {target_name}: {text}"

        start_time = time.time()

        translation = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=256
        )

        end_time = time.time()
        duration = end_time - start_time

        # Estimate tokens
        tokens = len(translation.split()) + len(text.split())

        return {
            "translation": translation.strip(),
            "time": duration,
            "tokens": tokens,
            "metadata": {
                "tokens_per_second": tokens / duration if duration > 0 else 0
            }
        }

    def get_backend_info(self) -> Dict[str, str]:
        """Get MLX backend info"""
        try:
            import mlx
            version = mlx.__version__
        except:
            version = "unknown"

        return {
            "name": "MLX",
            "version": version,
            "device": "metal",
            "model": self.model_id
        }
