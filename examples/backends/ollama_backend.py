"""Ollama backend for TranslateGemma"""
import time
import os
from typing import Dict, Any

try:
    from .base import TranslationBackend
except ImportError:
    from base import TranslationBackend


class OllamaBackend(TranslationBackend):
    """Ollama backend for local inference"""

    def __init__(self):
        super().__init__()
        self.model_name = "translategemma:latest"
        self.base_url = "http://localhost:11434"

    def load_model(self, **kwargs) -> Dict[str, Any]:
        """Check if Ollama is running and model is available"""
        import requests

        start_time = time.time()

        try:
            # Check Ollama server
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code != 200:
                raise RuntimeError("Ollama server not running")

            # Check if model is available
            models = response.json().get("models", [])
            model_names = [m["name"] for m in models]

            if not any("translategemma" in name for name in model_names):
                raise RuntimeError(
                    "TranslateGemma not found. Run: ollama pull translategemma"
                )

            load_time = time.time() - start_time

            return {
                "model_loaded": True,
                "load_time": load_time,
                "metadata": {
                    "backend": "ollama",
                    "model": self.model_name
                }
            }

        except Exception as e:
            raise RuntimeError(f"Ollama error: {e}")

    def translate(self, text: str, source_lang: str = "en", target_lang: str = "zh-TW") -> Dict[str, Any]:
        """Translate using Ollama"""
        import requests

        prompt = f"Translate from {source_lang} to {target_lang}:\n\n{text}"

        start_time = time.time()

        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0, "num_predict": 512}
            },
            timeout=120
        )

        result = response.json()
        translation = result.get("response", "").strip()
        duration = time.time() - start_time

        # Convert to Traditional Chinese
        if target_lang == "zh-TW":
            try:
                from hanziconv import HanziConv
                translation = HanziConv.toTraditional(translation)
            except:
                pass

        return {
            "translation": translation,
            "time": duration,
            "tokens": result.get("eval_count", 0),
            "metadata": {"tokens_per_second": result.get("eval_count", 0) / duration if duration > 0 else 0}
        }

    def get_backend_info(self) -> Dict[str, str]:
        return {"name": "Ollama", "model": self.model_name}

    def cleanup(self):
        pass
