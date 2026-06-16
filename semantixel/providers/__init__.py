"""Model provider implementations and registration infrastructure.

Every provider registers itself via the :func:`~.registry.provider`
decorator so that :class:`~semantixel.services.model_manager.ModelManager`
can resolve implementations by name without hard-coded if/elif chains.
"""

# Import concrete providers to trigger @provider decorator registration.
from semantixel.providers.clip import hf_provider  # noqa: F401
from semantixel.providers.ocr import doctr_provider  # noqa: F401
from semantixel.providers.text import hf_provider  # noqa: F401
from semantixel.providers.audio import clap_provider  # noqa: F401
from semantixel.providers.audio import hf_audio_provider  # noqa: F401
from semantixel.providers.audio import faster_whisper_provider  # noqa: F401
