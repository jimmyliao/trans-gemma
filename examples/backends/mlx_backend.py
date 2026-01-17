"""MLX backend for TranslateGemma (Apple Silicon optimized)"""
import time
from typing import Dict, Any
from .base import TranslationBackend


class MLXBackend(TranslationBackend):
    """MLX backend for Apple Silicon"""

    def __init__(self):
        super().__init__()
        self.model_id = "mlx-community/translategemma-4b-it"

    def load_model(self, **kwargs) -> Dict[str, Any]:
        """Load model using MLX"""
        try:
            from mlx_lm import load
        except ImportError:
            return {
                "model_loaded": False,
                "load_time": 0,
                "metadata": {"error": "mlx_lm not installed. Run: uv pip install mlx-lm"}
            }

        start_time = time.time()

        self.model, self.tokenizer = load(self.model_id)

        load_time = time.time() - start_time

        return {
            "model_loaded": True,
            "load_time": load_time,
            "metadata": {
                "model_id": self.model_id,
                "backend": "mlx",
                "device": "metal"
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
