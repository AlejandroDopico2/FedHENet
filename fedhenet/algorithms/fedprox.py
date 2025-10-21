import copy
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader

from .fedavg import FedAvg


class FedProx(FedAvg):
    def __init__(self, name: str = "fedprox", device: str = "cpu", **kwargs: Any):
        super().__init__(name=name, device=device, **kwargs)
        self.mu = kwargs.get("mu", 0.01)  # FedProx regularization parameter
        logger.debug(f"[FedProx DEBUG] mu={self.mu}")

    def local_train(
        self, model: nn.Module, loader: DataLoader, epochs: int = 1, **kwargs: Any
    ) -> nn.Module:
        """Train the model locally with FedProx regularization."""

        optimizer = torch.optim.SGD(
            model.fc.parameters(),
            lr=kwargs.get("learning_rate", 0.01),
            momentum=kwargs.get("momentum", 0.9),
        )

        global_model = copy.deepcopy(model).to(self.device)
        global_model.eval()

        model = model.to(self.device)
        model.fc.train()

        total_loss = 0.0
        num_batches = 0

        for _ in range(epochs):
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                preds = model(x)
                loss = self.criterion(preds, y)

                prox_reg = 0.0
                for (name, w) in model.named_parameters():
                    if not w.requires_grad:
                        continue
                    w_t = dict(global_model.named_parameters())[name]
                    w_t = w_t.detach().to(w.device)
                    prox_reg += ((w - w_t) ** 2).sum()
                
                loss = loss + (self.mu / 2) * prox_reg

                loss.backward()
                
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1


        # Move model back to CPU to save memory
        model.to("cpu")
        global_model.to("cpu")
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

        return model
