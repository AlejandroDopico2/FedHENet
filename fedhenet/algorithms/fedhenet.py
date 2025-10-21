# algorithms/fedheonn.py
from __future__ import annotations
from typing import Any, Dict, List, Optional

import base64
import pickle
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
import tenseal as ts
from loguru import logger

from ..rolann import ROLANN
from .base import BaseAlgorithm


class FedHENet(BaseAlgorithm):
    """
    FedHENet algorithm wrapper implementing your FedHENet / ROLANN logic as a BaseAlgorithm.
    - local_train: iterate local loader, extract features with extractor and call rolann.aggregate_update
    - compute_update: returns mg (list) and US (list) matrices (raw tensors or CKKS vectors)
    - serialize_update: transforms compute_update output to serializable envelope (base64/pickle or ckks)
    - aggregate_updates: classmethod that takes a list of client updates (rehydrated: ckks vectors or tensors)
                        and returns aggregated mg/ug/sg lists (same formats as inputs).
    - apply_global: apply aggregated update to the local rolann (set mg/ug/sg and recalc weights)
    """

    def __init__(
        self,
        name: str = "fedhenet",
        device: str = "cpu",
        **kwargs: Any,
    ) -> tuple[nn.Module, ROLANN]:
        super().__init__(name=name, device=device, **kwargs)

    def init_model(
        self, extractor: Optional[nn.Module] = None
    ) -> tuple[nn.Module, ROLANN]:

        rolann = ROLANN(num_classes=self.num_classes)
        if extractor is None:
            extractor = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            extractor.fc = nn.Identity()
            for p in extractor.parameters():
                p.requires_grad = False
        extractor = extractor.eval().to("cpu")
        rolann.to("cpu")

        return (extractor, rolann)

    def _process_label(self, num_classes, y):
        return torch.nn.functional.one_hot(y, num_classes=num_classes) * 0.9 + 0.05

    def local_train(
        self, model: nn.Module, loader: torch.utils.data.DataLoader, epochs: int = 1
    ) -> nn.Module:
        """
        In FedHeONN there is no optimizer training: we extract features and call rolann.aggregate_update
        - model param is ignored (kept for BaseAlgorithm compatibility)
        """
        # Move to device

        extractor, rolann = model
        extractor.to(self.device)
        rolann.to(self.device)

        # iterate over loader and update rolann
        with torch.no_grad():
            for x, y in loader:
                x = x.to(self.device)
                y = self._process_label(rolann.num_classes, y).to(self.device)
                feats = extractor(x)
                rolann.aggregate_update(feats, y)

        extractor.to("cpu")
        rolann.to("cpu")

        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

        return (extractor, rolann)

    def tensor_size_bytes(self, tensor):
        return tensor.element_size() * tensor.numel()

    def get_total_size_bytes(self, local_M, local_US):
        total = 0
        for t in list(local_M) + local_US:
            total += self.tensor_size_bytes(t)
        return total

    def compute_update(self, model: nn.Module) -> Dict[str, Any]:
        """
        Return the raw M and US lists (not yet serialized).
        M entries are either CKKS vectors (if encrypted) or torch.Tensor (float32).
        US entries are torch.Tensor (float32) matrices representing U @ diag(S) (same as your prior code).
        """
        local_M = model.mg
        local_US = [
            torch.matmul(model.ug[i], torch.diag(model.sg[i].clone().detach()))
            for i in range(model.num_classes)
        ]
        total_size_bytes = self.get_total_size_bytes(local_M, local_US)
        return {"m": local_M, "us": local_US, "metadata": {"algorithm": "fedhenet"}}

    def serialize_update(
        self, update: Dict[str, Any], encrypted: bool = False, ctx=None
    ) -> Dict[str, Any]:
        if encrypted:
            return self._serialize_ckks(update, ctx)
        else:
            return self._serialize_plain(update)

    def _serialize_ckks(
        self, update: Dict[str, Any], ctx: Optional[ts.Context] = None
    ) -> Dict[str, Any]:
        payload = []
        # TODO: Integrate float16
        for M_item, US in zip(update["m"], update["us"]):
            M_item = ts.ckks_vector(ctx, M_item.detach().cpu().numpy())
            bM = base64.b64encode(M_item.serialize()).decode()

            us_pickled = pickle.dumps(US.cpu().numpy())
            bUS_bytes = base64.b64encode(us_pickled)
            bUS = bUS_bytes.decode()
            payload.append({"m": bM, "us": bUS})

        envelope = {
            "version": 1,
            "format": "ckks",
            "num_classes": len(payload),
            "payload": payload,
            "metadata": update.get("metadata", {}),
        }
        return envelope

    def _serialize_plain(self, update: Dict[str, Any]) -> Dict[str, Any]:
        payload = []

        for M_item, US in zip(update["m"], update["us"]):
            m_pickled = pickle.dumps(M_item.cpu().numpy())
            bM_bytes = base64.b64encode(m_pickled)
            bM = bM_bytes.decode()

            us_pickled = pickle.dumps(US.cpu().numpy())
            bUS_bytes = base64.b64encode(us_pickled)
            bUS = bUS_bytes.decode()
            payload.append({"m": bM, "us": bUS})

        envelope = {
            "version": 1,
            "format": "plain",
            "num_classes": len(payload),
            "payload": payload,
            "metadata": update.get("metadata", {}),
        }
        return envelope

    def aggregate_updates(self, updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate a list of client updates and return aggregated mg/ug/sg lists.
        Each update is expected to be in *rehydrated* form (i.e. CKKS vectors or torch.Tensor objects),
        with structure: {'mg': [...], 'US': [...], 'metadata': {...}}
        """
        if not updates:
            return {}

        # number of classes
        nclasses = len(updates[0][1]["m"])
        device = torch.device("cpu")

        M_global, U_global, S_global = [], [], []

        for c in range(nclasses):
            M_acc, US_acc = None, None

            for upd in updates:
                M_k, US_k = upd[1]["m"][c], upd[1]["us"][c]

                M_acc = M_k if M_acc is None else M_acc + M_k

                US_k = US_k.to(device)
                if US_acc is None:
                    US_acc = US_k
                else:
                    concatenated = torch.cat((US_k, US_acc), dim=1)
                    U, S, _ = torch.linalg.svd(concatenated, full_matrices=False)
                    US_acc = U @ torch.diag(S)

            U_final, S_final, _ = torch.linalg.svd(US_acc, full_matrices=False)
            M_global.append(M_acc)
            U_global.append(U_final)
            S_global.append(S_final)

        return {
            "m": M_global,
            "u": U_global,
            "s": S_global,
            "metadata": {"algorithm": "fedhenet", "encrypted": self.encrypted},
        }

    def apply_global(
        self, model: nn.Module, aggregated_update: Dict[str, Any]
    ) -> nn.Module:
        if "m" not in aggregated_update:
            logger.warning("[FedHENet] No weights found in aggregated update")
            return model

        if self.encrypted:
            aggregated_update["m"] = [
                torch.FloatTensor(m.decrypt()) for m in aggregated_update["m"]
            ]

        model.mg = nn.ParameterList(
            [nn.Parameter(m, requires_grad=False) for m in aggregated_update["m"]]
        )
        model.ug = nn.ParameterList(
            [nn.Parameter(u, requires_grad=False) for u in aggregated_update["u"]]
        )
        model.sg = nn.ParameterList(
            [nn.Parameter(s, requires_grad=False) for s in aggregated_update["s"]]
        )
        model._calculate_weights()
        return model

    def deserialize_update(
        self,
        data: Dict[str, Any],
        global_update: bool = False,
    ) -> Dict[str, Any]:
        """
        Reconstruct mg and US lists from a serialized FedHENet update (plain or ckks).
        Returns:
            {
            "m": [...],   # list of torch.Tensors or ts.ckks_vector
            "us": [...],   # list of torch.Tensors
            "metadata": {...}
            }
        """
        if self.encrypted:
            return self._deserialize_ckks(data, global_update)
        else:
            return self._deserialize_plain(data, global_update)

    def _deserialize_plain(
        self, data: Dict[str, Any], global_update: bool = False
    ) -> Dict[str, Any]:
        payload = data.get("payload", [])
        mg_list = []

        if global_update:
            U_list, S_list = [], []
        else:
            US_list = []

        for entry in payload:
            M_np = pickle.loads(base64.b64decode(entry["m"]))

            if global_update:
                U_np = pickle.loads(base64.b64decode(entry["u"]))
                S_np = pickle.loads(base64.b64decode(entry["s"]))
            else:
                US_np = pickle.loads(base64.b64decode(entry["us"]))

            mg_list.append(torch.from_numpy(M_np))
            if global_update:
                U_list.append(torch.from_numpy(U_np))
                S_list.append(torch.from_numpy(S_np))
            else:
                US_list.append(torch.from_numpy(US_np))

        envelope = {
            "m": mg_list,
            "metadata": data.get("metadata", {}),
        }
        if global_update:
            envelope["u"] = U_list
            envelope["s"] = S_list
        else:
            envelope["us"] = US_list

        return envelope

    def _deserialize_ckks(
        self,
        data: Dict[str, Any],
        global_update: bool = False,
    ) -> Dict[str, Any]:
        if self.ctx is None and global_update:
            raise ValueError(
                "CKKS context required for encrypted deserialization on client"
            )

        payload = data.get("payload", [])
        mg_list = []
        if global_update:
            U_list, S_list = [], []
        else:
            US_list = []

        for entry in payload:
            # Rehydrate encrypted M
            M_bytes = base64.b64decode(entry["m"])

            if self.ctx:
                M = ts.ckks_vector_from(self.ctx, M_bytes)
            else:
                M = pickle.loads(M_bytes)
            mg_list.append(M)

            if global_update:
                U_np = pickle.loads(base64.b64decode(entry["u"]))
                S_np = pickle.loads(base64.b64decode(entry["s"]))
                U_list.append(torch.from_numpy(U_np))
                S_list.append(torch.from_numpy(S_np))
            else:
                US_np = pickle.loads(base64.b64decode(entry["us"]))
                US_list.append(torch.from_numpy(US_np))

        envelope = {
            "m": mg_list,
            "metadata": data.get("metadata", {}),
        }
        if global_update:
            envelope["u"] = U_list
            envelope["s"] = S_list
        else:
            envelope["us"] = US_list

        return envelope

    def serialize_global(
        self, global_update: Dict[str, Any], encrypted: bool = False
    ) -> Dict[str, Any]:
        """
        Serialize the aggregated global model (m, u, s) for broadcast.
        Supports both plain and CKKS encryption.
        """
        payload = []
        mg_list = global_update["m"]
        ug_list = global_update["u"]
        sg_list = global_update["s"]

        for M_global, U_global, S_global in zip(mg_list, ug_list, sg_list):
            # Serialize M
            if encrypted:
                if hasattr(M_global, "serialize"):
                    m_bytes = M_global.serialize()
                else:
                    m_bytes = pickle.dumps(M_global.cpu().numpy())
            else:
                m_bytes = pickle.dumps(M_global.cpu().numpy())

            # Serialize U and S separately
            u_bytes = pickle.dumps(U_global.cpu().numpy())
            s_bytes = pickle.dumps(S_global.cpu().numpy())

            payload.append(
                {
                    "m": base64.b64encode(m_bytes).decode(),
                    "u": base64.b64encode(u_bytes).decode(),
                    "s": base64.b64encode(s_bytes).decode(),
                }
            )

        return {
            "version": 1,
            "format": "ckks" if encrypted else "plain",
            "num_classes": len(payload),
            "payload": payload,
            "metadata": global_update.get("metadata", {}),
        }
