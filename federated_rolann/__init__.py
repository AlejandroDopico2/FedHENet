# federated_rolann/__init__.py
#  This file initializes the federated_rolann package and imports key components.

# Define the version of the package
__version__ = "1.0.0"

# Import the main ROLANN class from the core module
from .core import ROLANN

# Import the Client class from the federated.client module
from .federated.client import Client

# Import the Coordinator class from the federated.coordinator module
from .federated.coordinator import Coordinator

# Define the public API of the package
__all__ = ["ROLANN", "Client", "Coordinator"]
