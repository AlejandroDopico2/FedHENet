"""
Algorithm factory for creating federated learning algorithms by name.
"""

from typing import Any, Dict, Type

from .base import BaseAlgorithm
from .fedavg import FedAvg
from .fedprox import FedProx
from .fedhenet import FedHENet

# Registry of available algorithms
ALGORITHM_REGISTRY: Dict[str, Type[BaseAlgorithm]] = {
    "fedavg": FedAvg,
    "fedprox": FedProx,
    "fedhenet": FedHENet,
}


def get_algorithm(**kwargs: Any) -> BaseAlgorithm:
    """
    Create an algorithm instance by name.

    Args:
        name: Algorithm name (e.g., 'fedavg', 'fedprox', 'fedhenet')
        **kwargs: Algorithm-specific parameters

    Returns:
        Algorithm instance

    Raises:
        ValueError: If algorithm name is not supported

    Example:
        # Create FedAvg algorithm
        algorithm = get_algorithm("fedavg", device="cuda")

        # Create FedProx algorithm with mu parameter
        algorithm = get_algorithm("fedprox", device="cuda", mu=0.01)
    """
    name = kwargs.get("name", "fedhenet")
    if name not in ALGORITHM_REGISTRY:
        available = ", ".join(ALGORITHM_REGISTRY.keys())
        raise ValueError(
            f"Unknown algorithm '{name}'. Available algorithms: {available}"
        )

    defaults = get_algorithm_parameters(name)
    params = {**defaults, **kwargs}

    algorithm_class = ALGORITHM_REGISTRY[name]
    return algorithm_class(**params)  # type: ignore


def list_available_algorithms() -> list[str]:
    """
    Get list of available algorithm names.

    Returns:
        List of algorithm names
    """
    return list(ALGORITHM_REGISTRY.keys())


def get_algorithm_parameters(algorithm_name: str) -> Dict[str, Any]:
    """
    Get default parameters for a specific algorithm.

    Args:
        algorithm_name: Name of the algorithm

    Returns:
        Dictionary of default parameters for the algorithm
    """
    defaults = {
        "fedavg": {
            "name": "fedavg",
            "device": "cpu",
            "learning_rate": 0.01,
            "momentum": 0.9,
        },
        "fedprox": {
            "name": "fedprox",
            "device": "cpu",
            "learning_rate": 0.01,
            "momentum": 0.9,
            "mu": 0.01,  # FedProx regularization parameter
        },
        "fedhenet": {
            "name": "fedhenet",
            "device": "cpu",
            "encrypted": False,
            "ctx": None,
            "num_classes": 10,  # TODO: Fix
        },
    }

    return defaults.get(algorithm_name, {})


__all__ = [
    "get_algorithm",
    "list_available_algorithms",
    "get_algorithm_parameters",
    "ALGORITHM_REGISTRY",
]
