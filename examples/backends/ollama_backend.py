"""Ollama backend for TranslateGemma"""
import subprocess
import time
from typing import Dict, Any

try:
    from .base import TranslationBackend
except ImportError:
    from base import TranslationBackend


class OllamaBackend(TranslationBackend):
    """Ollama backend using ollama CLI"""

    def __init__(self):
        super().__init__()
        self.model_name = "translategemma:latest"

    def load_model(self, **kwargs) -> Dict[str, Any]:
        """Check if Ollama model is available"""
        start_time = time.time()

        try:
            # Check if ollama is installed
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                check=True
            )

            # Check if translategemma is installed
            if "translategemma" not in result.stdout:
                print("⚠️  TranslateGemma not found in Ollama")
                print("   Run: ollama pull translategemma")
                return {
                    "model_loaded": False,
                    "load_time": 0,
                    "metadata": {"error": "Model not found"}
                }

            load_time = time.time() - start_time
            self.model = True  # Mark as loaded

            return {
                "model_loaded": True,
                "load_time": load_time,
                "metadata": {
                    "model_name": self.model_name,
                    "backend": "ollama",
                    "device": "metal"  # Ollama uses Metal on M1
                }
            }

        except FileNotFoundError:
            return {
                "model_loaded": False,
                "load_time": 0,
                "metadata": {"error": "Ollama not installed"}
            }
        except Exception as e:
            return {
                "model_loaded": False,
                "load_time": 0,
                "metadata": {"error": str(e)}
            }

    def translate(
        self,
        text: str,
        source_lang: str = "en",
        target_lang: str = "zh-TW"
    ) -> Dict[str, Any]:
        """Translate using Ollama CLI"""

        # Map language codes to full names for better prompt
        lang_names = {
            "en": "English",
            "zh-TW": "Traditional Chinese",
            "zh-CN": "Simplified Chinese",
            "ja": "Japanese",
            "ko": "Korean",
            "es": "Spanish",
            "fr": "French",
            "de": "German"
        }

        target_name = lang_names.get(target_lang, target_lang)
        prompt = f"Translate this to {target_name}: {text}"

        start_time = time.time()

        try:
            result = subprocess.run(
                ["ollama", "run", self.model_name, prompt],
                capture_output=True,
                text=True,
                check=True
            )

            translation = result.stdout.strip()
            # Remove ANSI escape codes if any
            import re
            translation = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', translation)
            translation = re.sub(r'\[\?.*?[a-zA-Z]', '', translation)
            translation = translation.strip()

            end_time = time.time()
            duration = end_time - start_time

            # Estimate tokens (rough approximation)
            tokens = len(translation.split()) + len(text.split())

            return {
                "translation": translation,
                "time": duration,
                "tokens": tokens,
                "metadata": {
                    "tokens_per_second": tokens / duration if duration > 0 else 0,
                    "prompt": prompt
                }
            }

        except Exception as e:
            return {
                "translation": "",
                "time": 0,
                "tokens": 0,
                "metadata": {"error": str(e)}
            }

    def get_backend_info(self) -> Dict[str, str]:
        """Get Ollama backend info"""
        try:
            result = subprocess.run(
                ["ollama", "--version"],
                capture_output=True,
                text=True
            )
            version = result.stdout.strip() if result.returncode == 0 else "unknown"
        except:
            version = "unknown"

        return {
            "name": "Ollama",
            "version": version,
            "device": "metal",
            "model": self.model_name
        }
