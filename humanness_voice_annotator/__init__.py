"""One auditable voice-annotation format, backed by four published models."""
__version__ = "1.0.0"
from .formatting import compose_annotation, verify_script

__all__ = ["compose_annotation", "verify_script"]
