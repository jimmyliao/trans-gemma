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

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=256,
                do_sample=False
            )

        end_time = time.time()
        duration = end_time - start_time

        # Decode
        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract translation (after last newline or colon)
        translation = full_output.split('\n')[-1].strip()
        if ':' in translation:
            translation = translation.split(':', 1)[1].strip()

        # Post-processing: Convert Simplified to Traditional Chinese if needed
        if target_lang == "zh-TW":
            try:
                from hanziconv import HanziConv
                translation = HanziConv.toTraditional(translation)
            except ImportError:
                # hanziconv not installed, skip conversion
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
