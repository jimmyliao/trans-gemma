"""Transformers backend for TranslateGemma"""
import time
import os
from typing import Dict, Any

try:
    from .base import TranslationBackend
except ImportError:
    from base import TranslationBackend


class TransformersBackend(TranslationBackend):
    """Hugging Face Transformers backend"""

    def __init__(self):
        super().__init__()
        self.device_map = None
        self.torch_dtype = None

    def load_model(self, **kwargs) -> Dict[str, Any]:
        """Load model using transformers"""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import psutil

        start_time = time.time()

        # Get available memory
        mem = psutil.virtual_memory()
        available_mem_gb = mem.available / (1024**3)

        # Device selection logic
        force_device = os.getenv("FORCE_DEVICE")
        no_mem_limit = os.getenv("NO_MEM_LIMIT", "0") == "1"

        if force_device == "cpu":
            self.device_map = "cpu"
            self.torch_dtype = torch.float32
        elif force_device == "mps":
            self.device_map = "mps"
            self.torch_dtype = torch.bfloat16
        elif available_mem_gb < 8:
            self.device_map = "auto"
            self.torch_dtype = torch.bfloat16
        else:
            self.device_map = "auto"
            self.torch_dtype = torch.bfloat16

        # Prepare load kwargs
        load_kwargs = {
            "torch_dtype": self.torch_dtype,
            "low_cpu_mem_usage": True
        }

        # MPS memory limit
        if self.device_map == "auto" and torch.backends.mps.is_available() and not no_mem_limit:
            max_mem_gb = 8
            load_kwargs["max_memory"] = {0: f"{max_mem_gb}GiB", "cpu": "8GiB"}
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["device_map"] = self.device_map

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            **load_kwargs
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        load_time = time.time() - start_time

        # Get actual device
        device_info = f"{self.model.device}, dtype: {self.model.dtype}"

        return {
            "model_loaded": True,
            "load_time": load_time,
            "metadata": {
                "device": str(self.model.device),
                "dtype": str(self.model.dtype),
                "device_map": self.device_map,
                "available_memory_gb": available_mem_gb
            }
        }

    def translate(
        self,
        text: str,
        source_lang: str = "en",
        target_lang: str = "zh-TW"
    ) -> Dict[str, Any]:
        """Translate using transformers"""
        import torch

        # Build structured message
        messages = [{
            "role": "user",
            "content": [{
                "type": "text",
                "text": text,
                "source_lang_code": source_lang,
                "target_lang_code": target_lang
            }]
        }]

        # Apply chat template
        inputs = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt"
        ).to(self.model.device)

        start_time = time.time()

        # Generate (increased max_new_tokens to avoid truncation)
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=2048,  # Increased from 256 to avoid truncation
                do_sample=False
            )

        end_time = time.time()
        duration = end_time - start_time

        # Decode
        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Debug: Print full output if DEBUG env var is set
        import os
        if os.getenv('TRANSLATE_DEBUG'):
            print(f"\n{'='*80}")
            print(f"FULL OUTPUT ({len(full_output)} chars):")
            print(f"{'='*80}")
            print(full_output[:500])  # First 500 chars
            print(f"\n... [truncated] ...\n")
            print(full_output[-500:])  # Last 500 chars
            print(f"{'='*80}\n")

        # Extract translation with robust multi-strategy logic
        translation = self._extract_translation(full_output, source_lang, target_lang)

        # Debug: Print extracted translation
        if os.getenv('TRANSLATE_DEBUG'):
            print(f"EXTRACTED TRANSLATION ({len(translation)} chars):")
            print(translation[:200])
            print(f"\n{'='*80}\n")

        # Post-processing: Convert Simplified to Traditional Chinese if needed
        if target_lang == "zh-TW":
            try:
                from opencc import OpenCC
                cc = OpenCC('s2twp')  # Simplified to Traditional (Taiwan phrases)
                translation = cc.convert(translation)
            except ImportError:
                # Fallback to hanziconv if OpenCC not available
                try:
                    from hanziconv import HanziConv
                    translation = HanziConv.toTraditional(translation)
                except ImportError:
                    # Neither installed, skip conversion
                    pass

        # Calculate tokens
        input_tokens = inputs.shape[1]
        output_tokens = outputs.shape[1] - input_tokens
        total_tokens = outputs.shape[1]

        return {
            "translation": translation,
            "time": duration,
            "tokens": total_tokens,
            "metadata": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "tokens_per_second": total_tokens / duration if duration > 0 else 0
            }
        }

    def _extract_translation(self, full_output: str, source_lang: str, target_lang: str) -> str:
        """
        Extract translation from model output using multiple fallback strategies

        Args:
            full_output: Full decoded output from model
            source_lang: Source language code
            target_lang: Target language code

        Returns:
            Extracted translation text
        """
        import re

        # Strategy 1: Remove prompt template if present
        # TranslateGemma sometimes includes the full prompt in output
        if 'user' in full_output or 'You are a professional' in full_output:
            # Pattern 1: Remove everything up to the actual translation
            patterns = [
                r'user\n.*?(?:Please translate.*?:?\s*\n+)',  # user\n...Please translate...:
                r'You are a professional.*?(?:into|to).*?:?\s*\n+',  # You are a professional...into Chinese:
                r'^.*?Please translate the following.*?:?\s*\n+',  # Please translate the following...:
            ]

            for pattern in patterns:
                match = re.search(pattern, full_output, re.DOTALL | re.IGNORECASE)
                if match:
                    # Take everything after the match
                    candidate = full_output[match.end():].strip()
                    # Remove the source text if it appears at the start
                    lines = candidate.split('\n')
                    if len(lines) > 1:
                        # Try to find where translation starts (usually after source text)
                        for i, line in enumerate(lines):
                            if self._looks_like_target_language(line, target_lang):
                                translation = '\n'.join(lines[i:]).strip()
                                if len(translation) > 10:
                                    return translation
                    elif self._looks_like_target_language(candidate, target_lang):
                        return candidate

        # Strategy 2: Split by double newline and find target language part
        if '\n\n' in full_output:
            parts = [p.strip() for p in full_output.split('\n\n') if p.strip()]
            # Try to find the part that looks like target language
            for part in reversed(parts):  # Start from the end
                if self._looks_like_target_language(part, target_lang) and len(part) > 10:
                    return part

        # Strategy 3: Check for labeled output (e.g., "Translation: xxx")
        if ':' in full_output:
            lines = full_output.split('\n')
            for i, line in enumerate(lines):
                if ':' in line and len(line.split(':', 1)) > 1:
                    label, content = line.split(':', 1)
                    # Check if label suggests this is the translation
                    if any(keyword in label.lower() for keyword in ['translation', 'output', 'result']):
                        # Take this line and all following lines
                        remaining = '\n'.join(lines[i:])
                        content = remaining.split(':', 1)[1].strip()
                        if len(content) > 10:
                            return content

        # Strategy 4: Take the last substantial line/paragraph
        lines = [l.strip() for l in full_output.split('\n') if l.strip()]
        if lines:
            # Get the last line that's substantial enough
            for line in reversed(lines):
                if len(line) > 10 and self._looks_like_target_language(line, target_lang):
                    return line
            # Fallback: just take the last line
            if len(lines[-1]) > 10:
                return lines[-1]

        # Last resort: return full output (cleaned)
        return full_output.strip()

    def _looks_like_target_language(self, text: str, lang_code: str) -> bool:
        """
        Heuristic to check if text appears to be in the target language

        Args:
            text: Text to check
            lang_code: Language code (e.g., 'zh-TW', 'en', 'ja')

        Returns:
            True if text appears to be in target language
        """
        if not text or len(text) < 3:
            return False

        # Chinese (Traditional or Simplified)
        if lang_code.startswith('zh'):
            # Check for CJK Unified Ideographs
            cjk_count = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
            return cjk_count > len(text) * 0.2  # At least 20% CJK

        # Japanese
        elif lang_code == 'ja':
            # Hiragana, Katakana, or Kanji
            jp_count = sum(1 for c in text if
                          ('\u3040' <= c <= '\u309f') or  # Hiragana
                          ('\u30a0' <= c <= '\u30ff') or  # Katakana
                          ('\u4e00' <= c <= '\u9fff'))    # Kanji
            return jp_count > len(text) * 0.15

        # Korean
        elif lang_code == 'ko':
            kr_count = sum(1 for c in text if '\uac00' <= c <= '\ud7af')  # Hangul
            return kr_count > len(text) * 0.2

        # English and Latin-script languages
        elif lang_code in ['en', 'de', 'fr', 'es', 'it', 'pt', 'nl']:
            ascii_count = sum(1 for c in text if ord(c) < 128)
            return ascii_count > len(text) * 0.7

        # Default: can't determine, assume it could be the target
        return True

    def get_backend_info(self) -> Dict[str, str]:
        """Get transformers backend info"""
        import transformers
        import torch

        return {
            "name": "Transformers",
            "version": transformers.__version__,
            "torch_version": torch.__version__,
            "device": str(self.device_map),
            "model": self.model_id
        }

    def cleanup(self):
        """Clean up model from memory"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
