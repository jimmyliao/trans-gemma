"""MLX backend for TranslateGemma (Apple Silicon optimized)"""
import time
from typing import Dict, Any
from .base import TranslationBackend


class MLXBackend(TranslationBackend):
    """MLX backend for Apple Silicon (4-bit quantized)

    ⚠️  EXPERIMENTAL: Translation quality may vary.
    MLX-optimized TranslateGemma requires specific prompt formatting.
    For production use, we recommend the Ollama backend which has been
    fully tested and validated.
    """

    def __init__(self):
        super().__init__()
        # Use 4-bit quantized version for better performance
        self.model_id = "mlx-community/translategemma-4b-it-4bit"

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

        try:
            self.model, self.tokenizer = load(self.model_id)
            load_time = time.time() - start_time

            return {
                "model_loaded": True,
                "load_time": load_time,
                "metadata": {
                    "model_id": self.model_id,
                    "backend": "mlx",
                    "device": "metal",
                    "quantization": "4-bit"
                }
            }
        except Exception as e:
            return {
                "model_loaded": False,
                "load_time": 0,
                "metadata": {"error": f"Failed to load MLX model: {str(e)}"}
            }

    def translate(
        self,
        text: str,
        source_lang: str = "en",
        target_lang: str = "zh-TW"
    ) -> Dict[str, Any]:
        """Translate using MLX"""
        from mlx_lm import generate

        # Map language codes to names
        lang_names = {
            "en": "English",
            "zh-TW": "Traditional Chinese (Taiwan)",
            "zh-CN": "Simplified Chinese",
            "ja": "Japanese",
            "ko": "Korean",
            "es": "Spanish",
            "fr": "French",
            "de": "German"
        }

        target_name = lang_names.get(target_lang, target_lang)

        # Use simple prompt format (similar to Ollama's success)
        prompt = f"Translate this to {target_name}: {text}"

        start_time = time.time()

        # Generate translation with limited tokens to avoid loops
        response = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=128,  # Reduced to avoid runaway generation
            verbose=False
        )

        end_time = time.time()
        duration = end_time - start_time

        # Clean up response - remove special tokens and extra whitespace
        translation = response.strip()

        # Remove common artifacts
        translation = translation.replace('<end_of_turn>', '')
        translation = translation.replace('<start_of_turn>', '')
        translation = translation.replace('user', '')
        translation = translation.replace('model', '')

        # Get first meaningful line
        lines = [line.strip() for line in translation.split('\n') if line.strip()]
        if lines:
            translation = lines[0]

        # Estimate actual tokens (exclude special tokens)
        tokens = len([t for t in response.split() if not t.startswith('<')])

        return {
            "translation": translation.strip(),
            "time": duration,
            "tokens": tokens if tokens > 0 else 1,
            "metadata": {
                "tokens_per_second": tokens / duration if duration > 0 and tokens > 0 else 0,
                "raw_response": response[:200] if len(response) > 200 else response
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
