# Translation backends for TranslateGemma
from .base import TranslationBackend
from .transformers_backend import TransformersBackend
from .transformers_multimodal_backend import TransformersMultimodalBackend
from .ollama_backend import OllamaBackend

try:
    from .mlx_backend import MLXBackend
    HAS_MLX = True
except ImportError:
    MLXBackend = None
    HAS_MLX = False

__all__ = [
    'TranslationBackend',
    'TransformersBackend',
    'TransformersMultimodalBackend',
    'OllamaBackend',
    'MLXBackend',
    'HAS_MLX',
    'get_backend'
]

def get_backend(name='transformers'):
    """Factory function to get translation backend"""
    backends = {
        'transformers': TransformersBackend,
        'ollama': OllamaBackend,
    }

    if HAS_MLX:
        backends['mlx'] = MLXBackend

    if name not in backends:
        available = ', '.join(backends.keys())
        raise ValueError(f"Unknown backend: {name}. Available: {available}")

    return backends[name]()
