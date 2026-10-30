"""Submit and retrieve ego-anno annotations (Python 3.8+, standard library only)."""
from .client import EgoAnnoClient
from .transport import ClientError, PinError, RetryableError

__version__ = "0.1.0"

__all__ = ["EgoAnnoClient", "ClientError", "PinError", "RetryableError", "__version__"]
