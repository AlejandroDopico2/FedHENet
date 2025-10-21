from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import torch.nn as nn
from torch.utils.data import DataLoader


class BaseAlgorithm(ABC):
    def __init__(self, name: str, device: str = "cpu", **kwargs: Any):
        self.name = name
        self.device = device
        self.encrypted = kwargs["encrypted"]
        self.ctx = kwargs["ctx"]
        self.num_classes = kwargs["num_classes"]

        self._last_global: Optional[Dict[str, Any]] = None

    def on_round_start(self, round_idx: int) -> None:
        """Optional: reset states, update learning rate scheduler, etc."""
        pass

    @abstractmethod
    def init_model(self, extractor: Optional[nn.Module] = None) -> nn.Module:
        """Initialize the model."""
        raise NotImplementedError

    @abstractmethod
    def local_train(
        self, model: nn.Module, loader: DataLoader, epochs: int = 1
    ) -> nn.Module:
        """Perform local training or feature extraction (FedHENet)."""
        raise NotImplementedError

    @abstractmethod
    def compute_update(self, model: nn.Module) -> Dict[str, Any]:
        """
        Returns the actualization that will be sent to the coordinator.

        Returns:
            Dictionary containing the actualization.
            Form: {'weights': [...], 'metadata': {}}
        """
        raise NotImplementedError

    @abstractmethod
    def serialize_update(
        self, update: Dict[str, Any], encrypted: bool, ctx=None
    ) -> Dict[str, Any]:
        """
        Serialize the update for network transmission.
        Each algorithm defines its own structure (e.g., weights, stats).
        """
        pass

    @abstractmethod
    def aggregate_updates(self, updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Logic to aggregate multiple updates in the Coordinator.
        Returns the aggregated update that will be used to construct the new global.
        """
        raise NotImplementedError

    @abstractmethod
    def apply_global(
        self, model: nn.Module, aggregated_update: Dict[str, Any]
    ) -> nn.Module:
        raise NotImplementedError

    @abstractmethod
    def deserialize_update(
        self,
        data: Dict[str, Any],
        global_update: bool = False,
    ) -> Dict[str, Any]:
        """
        Deserialize the update from network transmission.
        Each algorithm defines its own structure (e.g., weights, stats).
        """
        pass

    def deserialize_global(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deserialize the global model from network transmission.
        Each algorithm defines its own structure (e.g., weights, stats).
        """
        pass

    @abstractmethod
    def serialize_global(self, global_update: Dict[str, Any]) -> Dict[str, Any]:
        """
        Serialize the update for network transmission.
        Each algorithm defines its own structure (e.g., weights, stats).
        """
        pass

    def save_state(self) -> Dict:
        """Serialize local state (e.g. control variates)."""
        return {}

    def load_state(self, state: Dict) -> None:
        """Restore state from checkpoint."""
        pass
