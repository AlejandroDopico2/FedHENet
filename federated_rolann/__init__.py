__version__ = "1.0.0"

from .federated.client import Client
from .federated.coordinator import Coordinator

__all__ = ["Client", "Coordinator"]
