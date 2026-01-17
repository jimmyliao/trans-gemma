"""Transformers Multimodal backend for TranslateGemma (Image support)"""
import time
import os
from typing import Dict, Any, Union
from pathlib import Path
from .base import TranslationBackend


class TransformersMultimodalBackend(TranslationBackend):
    """Hugging Face Transformers backend with multimodal support

    Supports both text and image translation using TranslateGemma's
    image-text-to-text capabilities.

    ⚠️  EXPERIMENTAL: Image translation is slower but preserves layout context.
    """

    def __init__(self):
        super().__init__()
        self.device_map = None
        self.torch_dtype = None
        self.processor = None

    def load_model(self, **kwargs) -> Dict[str, Any]:
        """Load multimodal model using transformers"""
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
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

        # Load multimodal model
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            **load_kwargs
        )

        # Load processor (handles both text and images)
        self.processor = AutoProcessor.from_pretrained(self.model_id)

        load_time = time.time() - start_time

        return {
            "model_loaded": True,
            "load_time": load_time,
            "metadata": {
                "device": str(self.model.device),
                "dtype": str(self.model.dtype),
                "device_map": self.device_map,
                "available_memory_gb": available_mem_gb,
                "multimodal": True
            }
        }

    def translate(
        self,
        text: str,
        source_lang: str = "en",
        target_lang: str = "zh-TW"
    ) -> Dict[str, Any]:
        """Translate text using transformers (text mode)"""
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

        # Apply chat template using processor
        inputs = self.processor.apply_chat_template(
            messages,
            return_tensors="pt"
        ).to(self.model.device)

        start_time = time.time()

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.3,        # Lower for consistent language
                repetition_penalty=1.5,
                no_repeat_ngram_size=3,
                top_p=0.85,
                top_k=40
            )

        end_time = time.time()
        duration = end_time - start_time

        # Decode
        full_output = self.processor.decode(outputs[0], skip_special_tokens=True)

        # Extract translation (after last newline or colon)
        translation = full_output.split('\n')[-1].strip()
        if ':' in translation:
            translation = translation.split(':', 1)[1].strip()

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
                "tokens_per_second": total_tokens / duration if duration > 0 else 0,
                "mode": "text"
            }
        }

    def translate_image(
        self,
        image_path: Union[str, Path],
        source_lang: str = "en",
        target_lang: str = "zh-TW",
        stream: bool = True
    ) -> Dict[str, Any]:
        """Translate text from image using multimodal capabilities

        Args:
            image_path: Path to image file or PIL Image
            source_lang: Source language code
            target_lang: Target language code
            stream: Enable streaming generation with early stopping (default: True)
                   - Provides real-time progress feedback
                   - Detects and stops repetition automatically
                   - Better user experience but same total time

        Returns:
            Dictionary with translation and metadata
        """
        import torch
        from PIL import Image

        # Load image
        if isinstance(image_path, (str, Path)):
            image = Image.open(image_path)
        else:
            image = image_path  # Assume it's already a PIL Image

        # Resize to TranslateGemma's expected input size (896x896)
        # This significantly speeds up processing
        original_size = image.size
        target_size = 896

        # Resize maintaining aspect ratio, then pad to square
        width, height = image.size
        if width > target_size or height > target_size:
            # Calculate resize ratio
            ratio = min(target_size / width, target_size / height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)

            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Pad to square (896x896)
            new_image = Image.new("RGB", (target_size, target_size), (255, 255, 255))
            paste_x = (target_size - new_width) // 2
            paste_y = (target_size - new_height) // 2
            new_image.paste(image, (paste_x, paste_y))
            image = new_image

        # Just use structured message format without extra instructions
        # TranslateGemma should understand the language codes directly
        messages = [{
            "role": "user",
            "content": [{
                "type": "image",
                "source_lang_code": source_lang,
                "target_lang_code": target_lang
            }]
        }]

        start_time = time.time()

        # Process inputs with image using processor
        # For multimodal models, we need to use the processor correctly
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Process both text and image
        inputs = self.processor(
            text=text,
            images=image,
            return_tensors="pt"
        )

        # Move to device
        inputs = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v
                 for k, v in inputs.items()}

        if stream:
            # Use streaming generation with early stopping
            from transformers import TextIteratorStreamer
            from threading import Thread
            import sys

            streamer = TextIteratorStreamer(
                self.processor.tokenizer,
                skip_prompt=True,
                skip_special_tokens=True
            )

            generation_kwargs = {
                **inputs,
                "max_new_tokens": 2048,
                "do_sample": True,
                "temperature": 0.3,        # Lower temperature for more consistent language (was 0.7)
                "repetition_penalty": 1.5,
                "no_repeat_ngram_size": 3,
                "top_p": 0.85,             # Slightly lower for more focused output (was 0.9)
                "top_k": 40,               # Reduce randomness (was 50)
                "streamer": streamer
            }

            # Start generation in background thread
            thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
            thread.start()

            # Collect output with early stopping detection
            full_output = ""
            last_chunk = ""
            repetition_count = 0
            max_repetition = 3  # Stop if same text repeats 3 times

            print("Translation (streaming):", end=" ", flush=True)

            for new_text in streamer:
                full_output += new_text
                print(new_text, end="", flush=True)

                # Early stopping: detect repetition
                if new_text.strip() and new_text.strip() == last_chunk.strip():
                    repetition_count += 1
                    if repetition_count >= max_repetition:
                        print("\n⚠️  Repetition detected, stopping early...", flush=True)
                        break
                else:
                    repetition_count = 0
                    last_chunk = new_text

            print()  # New line after streaming
            thread.join()

            # Extract translation
            translation = full_output.split('\n')[-1].strip()
            if ':' in translation:
                translation = translation.split(':', 1)[1].strip()

            # Calculate tokens (estimate from output length)
            output_tokens = len(self.processor.tokenizer.encode(full_output))

        else:
            # Original non-streaming generation
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    do_sample=True,
                    temperature=0.3,        # Lower for consistent language
                    repetition_penalty=1.5,
                    no_repeat_ngram_size=3,
                    top_p=0.85,
                    top_k=40
                )

            # Decode
            full_output = self.processor.decode(outputs[0], skip_special_tokens=True)

            # Extract translation
            translation = full_output.split('\n')[-1].strip()
            if ':' in translation:
                translation = translation.split(':', 1)[1].strip()

            output_tokens = outputs.shape[1]

        end_time = time.time()
        duration = end_time - start_time

        # Calculate tokens
        input_ids = inputs.get('input_ids', None)
        input_tokens = input_ids.shape[1] if input_ids is not None else 0

        if not stream:
            # Non-streaming: use output tensor shape
            total_tokens = outputs.shape[1]
            output_tokens = total_tokens - input_tokens if input_tokens > 0 else total_tokens
        else:
            # Streaming: output_tokens already calculated above
            total_tokens = input_tokens + output_tokens

        return {
            "translation": translation,
            "time": duration,
            "tokens": total_tokens,
            "metadata": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "tokens_per_second": total_tokens / duration if duration > 0 else 0,
                "mode": "image",
                "image_size": image.size,
                "original_size": original_size,
                "resized": original_size != image.size
            }
        }

    def get_backend_info(self) -> Dict[str, str]:
        """Get transformers multimodal backend info"""
        import transformers
        import torch

        return {
            "name": "Transformers (Multimodal)",
            "version": transformers.__version__,
            "torch_version": torch.__version__,
            "device": str(self.device_map),
            "model": self.model_id,
            "capabilities": "text + image"
        }

    def cleanup(self):
        """Clean up model from memory"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None

        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
