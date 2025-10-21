from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import base64
import pickle
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader
from torchvision import models
import tenseal as ts

from .base import BaseAlgorithm


class FedAvg(BaseAlgorithm):
    def __init__(self, name: str = "fedavg", device: str = "cpu", **kwargs: Any):
        super().__init__(name=name, device=device, **kwargs)
        self.criterion = nn.CrossEntropyLoss()

    def init_model(self, extractor: Optional[nn.Module] = None) -> nn.Module:
        if extractor is None:
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            for p in model.parameters():
                p.requires_grad = False

        model.fc = nn.Linear(model.fc.in_features, self.num_classes)

        model.eval()
        model.fc.train()

        model = model.to("cpu")

        return model

    def local_train(
        self, model: nn.Module, loader: DataLoader, epochs: int = 1, **kwargs: Any
    ) -> nn.Module:
        """Train the model locally on the provided dataset."""

        optimizer = torch.optim.SGD(
            model.fc.parameters(),
            lr=kwargs.get("learning_rate", 0.01),
            momentum=kwargs.get("momentum", 0.9),
        )

        criterion = self.criterion
        device = self.device

        model.to(self.device)
        model.fc.train()

        for _ in range(epochs):
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                preds = model(x)
                loss = criterion(preds, y)
                loss.backward()
                optimizer.step()

        # Move model back to CPU to save memory
        model.to("cpu")
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

        return model

    def compute_update(self, model: nn.Module) -> Dict[str, Any]:
        """Collect and send current trainable weights to the coordinator."""
        trainable_weights = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                # Send the full parameter tensor (weights/biases)
                trainable_weights[name] = param.data.clone()

        return {"weights": trainable_weights, "metadata": {"algorithm": "fedavg"}}

    def serialize_update(self, update: Dict[str, Any]) -> Dict[str, Any]:
        if self.encrypted:
            return self._serialize_ckks(update)
        else:
            return self._serialize_plain(update)

    def _serialize_ckks(self, update: Dict[str, Any]) -> Dict[str, Any]:
        payload = []
        for name, param in update["weights"].items():
            if isinstance(param, dict):
                tensor_shape = param["shape"]
                encrypted = param["cipher"]
            else:
                tensor_cpu = param.detach().cpu().numpy().astype(np.float32)
                tensor_shape = list(tensor_cpu.shape)
                flat = tensor_cpu.flatten().tolist()
                encrypted = ts.ckks_vector(self.ctx, flat)

            encrypted_data = {
                "data": base64.b64encode(encrypted.serialize()).decode(),
                "shape": tensor_shape,
            }
            payload.append({"name": name, "weights": encrypted_data})
        return {"payload": payload, "metadata": update.get("metadata", {})}

    def _serialize_plain(self, update: Dict[str, Any]) -> Dict[str, Any]:
        payload = []
        for name, param in update["weights"].items():
            w_bytes = base64.b64encode(pickle.dumps(param)).decode()
            payload.append({"name": name, "weights": w_bytes})
        return {"payload": payload, "metadata": update.get("metadata", {})}

    def apply_global(
        self, model: nn.Module, aggregated_update: Dict[str, Any]
    ) -> nn.Module:
        """Apply the global aggregated weights to the local model."""
        if "weights" in aggregated_update:
            for name, param in model.named_parameters():
                if param.requires_grad and name in aggregated_update["weights"]:
                    # Replace local parameters with the aggregated global weights
                    global_tensor = aggregated_update["weights"][name]
                    # Ensure same dtype and device as destination parameter
                    if isinstance(global_tensor, torch.Tensor):
                        if global_tensor.dtype != param.data.dtype:
                            global_tensor = global_tensor.to(dtype=param.data.dtype)
                        if global_tensor.device != param.data.device:
                            global_tensor = global_tensor.to(device=param.data.device)
                        param.data.copy_(global_tensor)
                    else:
                        # Fallback in unlikely case it's not a tensor (shouldn't happen in plaintext mode)
                        param.data.copy_(
                            torch.tensor(
                                global_tensor,
                                dtype=param.data.dtype,
                                device=param.data.device,
                            )
                        )
        else:
            logger.warning("[FedAvg] No weights found in aggregated update")

        return model

    def deserialize_update(
        self,
        data: Dict[str, Any],
        global_update: bool = False,
    ) -> Dict[str, Any]:

        if self.encrypted:
            return self._deserialize_ckks(data, global_update)
        else:
            return self._deserialize_plain(data, global_update)

    def _deserialize_plain(
        self, data: Dict[str, Any], global_update: bool = False
    ) -> Dict[str, Any]:
        payload = data.get("payload", [])
        weights = {}
        for entry in payload:
            weights[entry["name"]] = pickle.loads(base64.b64decode(entry["weights"]))
        return {"weights": weights, "metadata": data.get("metadata", {})}

    def _deserialize_ckks(
        self,
        data: Dict[str, Any],
        global_update: bool = False,
    ) -> Dict[str, Any]:
        payload = data.get("payload", [])
        weights = {}
        for entry in payload:
            if isinstance(entry["weights"], dict) and "data" in entry["weights"]:
                encrypted_bytes = base64.b64decode(entry["weights"]["data"])
                cipher = ts.ckks_vector_from(self.ctx, encrypted_bytes)

                if global_update:
                    flat_tensor = cipher.decrypt()
                    tensor = torch.FloatTensor(flat_tensor)
                    tensor = tensor.reshape(entry["weights"].get("shape", []))
                    weights[entry["name"]] = tensor
                else:
                    weights[entry["name"]] = {
                        "cipher": cipher,
                        "shape": entry["weights"].get("shape", []),
                    }
            else:
                weights[entry["name"]] = pickle.loads(
                    base64.b64decode(entry["weights"])
                )
        return {"weights": weights, "metadata": data.get("metadata", {})}

    def serialize_global(self, global_update: Dict[str, Any]) -> Dict[str, Any]:
        return self.serialize_update(global_update)

    def aggregate_updates(self, updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate multiple client updates using FedAvg rule.

        Args:
            updates: List of tuples (num_samples, update_dict) where update_dict contains 'weights'

        Returns:
            Aggregated weights dictionary
        """

        if not updates:
            logger.warning("[FedAvg] No updates to aggregate")
            return {}

        # Extract weights and sample counts
        weights_and_samples = []
        total_samples = 0
        for update in updates:
            if isinstance(update, tuple):
                num_samples, update_dict = update
            else:
                num_samples = update.get("metadata", {}).get("num_samples", 1)
                update_dict = update

            total_samples += num_samples

            if "weights" in update_dict:
                weights_and_samples.append((num_samples, update_dict["weights"]))

        if not weights_and_samples or total_samples == 0:
            logger.warning("[FedAvg] No valid weights found in updates")
            return {}

        # FedAvg aggregation
        aggregated_weights = self.fedavg_aggregate(weights_and_samples)

        return {
            "weights": aggregated_weights,
            "metadata": {"algorithm": "fedavg", "encrypted": self.encrypted},
        }

    def fedavg_aggregate(
        self, updates: List[Tuple[int, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """
        FedAvg aggregation rule for full weights (supports plaintext and CKKS encrypted).

        Args:
            updates: List of tuples (num_samples, weights_dict)
                - num_samples: number of training samples on the client
                - weights_dict: full trainable parameter tensors from local models
                    If encrypted=True -> {"cipher": CKKSVector, "shape": tuple}
                    If encrypted=False -> torch.Tensor

        Returns:
            Aggregated weights dictionary (trainable parameters only)
        """
        if not updates:
            return {}

        total_samples = float(sum(int(num_samples) for num_samples, _ in updates))
        if total_samples == 0:
            return {}

        # Initialize aggregated weights with first client's structure
        aggregated_weights = {}
        first_weights = updates[0][1]

        for key in first_weights.keys():
            acc = None
            shape = None

            for num_samples, weights in updates:
                num_samples = float(num_samples)
                rel_weight = float(num_samples / total_samples)

                if self.encrypted:
                    cipher = weights[key]["cipher"]
                    scaled = cipher * rel_weight

                    # Try to keep CKKS scales under control after scalar multiplication
                    # Some TenSEAL versions expose rescale_next (in-place) to normalize scale.
                    try:
                        if hasattr(scaled, "rescale_next"):
                            # Depending on version, this may be in-place or return a new vector
                            maybe = scaled.rescale_next()
                            if maybe is not None:
                                scaled = maybe
                    except Exception:
                        # Best-effort: if rescale isn't available, proceed without it
                        pass

                    if acc is None:
                        acc = scaled
                    else:
                        acc += scaled

                    shape = weights[key].get("shape")
                else:
                    np_array = weights[key]
                    scaled = rel_weight * np_array

                    if acc is None:
                        acc = scaled.clone()
                    else:
                        acc += scaled

            aggregated_weights[key] = (
                {"cipher": acc, "shape": shape} if self.encrypted else acc
            )

        return aggregated_weights
