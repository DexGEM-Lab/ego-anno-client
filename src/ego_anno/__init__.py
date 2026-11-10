"""Submit/retrieve annotations; optional extras add offline visualization."""
from .client import EgoAnnoClient
from .transport import ClientError, PinError, RetryableError

__version__ = "0.4.1"

__all__ = ["EgoAnnoClient", "ClientError", "PinError", "RetryableError", "__version__"]
