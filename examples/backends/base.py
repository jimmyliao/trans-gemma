"""Base class for translation backends"""
from abc import ABC, abstractmethod
from typing import Dict, Any


class TranslationBackend(ABC):
    """Abstract base class for translation backends"""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_id = "google/translategemma-4b-it"

    @abstractmethod
    def load_model(self, **kwargs) -> Dict[str, Any]:
        """Load model and return metadata

        Returns:
            Dict with keys: model_loaded (bool), load_time (float), metadata (dict)
        """
        pass

    @abstractmethod
    def translate(
        self,
        text: str,
        source_lang: str = "en",
        target_lang: str = "zh-TW"
    ) -> Dict[str, Any]:
        """Translate text from source to target language

        Args:
            text: Text to translate
            source_lang: Source language code (ISO 639-1)
            target_lang: Target language code (ISO 639-1)

        Returns:
            Dict with keys: translation (str), time (float), tokens (int), metadata (dict)
        """
        pass

    @abstractmethod
    def get_backend_info(self) -> Dict[str, str]:
        """Get backend information

        Returns:
            Dict with keys: name, version, device, etc.
        """
        pass

    def cleanup(self):
        """Optional cleanup method"""
        pass
