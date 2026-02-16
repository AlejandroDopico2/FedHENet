"""Base model abstractions for FedHENet."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Iterable, Mapping

import torch
import torch.nn as nn


class BaseFLModel(nn.Module, ABC):
    """Split-model interface used by FedHENet algorithms.

    The interface isolates the (often frozen) feature extractor from the
    trainable classification head.  Federated strategies only need to touch
    `get_trainable_parameters` and `set_trainable_parameters`, allowing us to
    keep transport and encryption logic agnostic to the concrete backbone.
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def get_feature_extractor(self) -> nn.Module:
        """Return the feature extractor component."""

    @abstractmethod
    def get_classifier(self) -> nn.Module:
        """Return the trainable classification head."""

    def classifier_parameters(self) -> Iterable[nn.Parameter]:
        """Convenience handle for optimizer creation."""

        return self.get_classifier().parameters()

    def get_trainable_parameters(self) -> Dict[str, torch.Tensor]:
        """Return a CPU copy of the classifier state dict."""

        return {
            name: tensor.detach().cpu().clone()
            for name, tensor in self.get_classifier().state_dict().items()
        }

    def set_trainable_parameters(
        self, parameters: Mapping[str, torch.Tensor]
    ) -> None:
        """Load classifier weights coming from the coordinator."""

        classifier = self.get_classifier()
        current_state = classifier.state_dict()
        missing = sorted(set(current_state) - set(parameters))
        extra = sorted(set(parameters) - set(current_state))
        if missing or extra:
            raise KeyError(
                "Incompatible classifier state. "
                f"Missing: {missing or 'None'}, Extra: {extra or 'None'}"
            )

        updated_state: Dict[str, torch.Tensor] = {}
        for name, tensor in parameters.items():
            ref_tensor = current_state[name]
            updated_state[name] = tensor.to(
                device=ref_tensor.device,
                dtype=ref_tensor.dtype,
            )

        classifier.load_state_dict(updated_state, strict=True)


class FeatureExtractorClassifierModel(BaseFLModel):
    """Concrete split-model composed of extractor + classifier."""

    def __init__(
        self,
        feature_extractor: nn.Module,
        classifier: nn.Module,
        *,
        flatten_features: bool = False,
    ) -> None:
        super().__init__()
        self._feature_extractor = feature_extractor
        self._classifier = classifier
        self._flatten_features = flatten_features

        # Temporary compatibility with pre-refactor code paths that used .fc
        self.fc = classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self._feature_extractor(x)
        if self._flatten_features and features.ndim > 2:
            features = torch.flatten(features, 1)
        return self._classifier(features)

    def get_feature_extractor(self) -> nn.Module:
        return self._feature_extractor

    def get_classifier(self) -> nn.Module:
        return self._classifier

