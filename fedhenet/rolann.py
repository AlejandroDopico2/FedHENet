# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 10:11:21 2024

@author: Oscar Fontenla & Alejandro Dopico
"""

from typing import List, Optional

import torch
import torch.nn as nn
from torch import Tensor

import tenseal as ts

from loguru import logger


class ROLANN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        context: ts.Context | None = None,
    ):
        super(ROLANN, self).__init__()

        self.num_classes = num_classes

        self.f = torch.sigmoid
        self.finv = lambda x: torch.log(x / (1 - x))
        self.fderiv = lambda x: x * (1 - x)

        self.m: Optional[List[Tensor]] = None
        self.u: Optional[List[Tensor]] = None
        self.s: Optional[List[Tensor]] = None
        self.m_original_shapes: Optional[List[tuple]] = None

        self.mg: nn.ParameterList = nn.ParameterList()
        self.ug: nn.ParameterList = nn.ParameterList()
        self.sg: nn.ParameterList = nn.ParameterList()
        self.w: nn.ParameterList = nn.ParameterList()

    def update_weights(self, X: Tensor, d: Tensor) -> None:
        """
        Computes M, U, and S for a batch of classes in parallel, removing the original loop.
        """
        num_samples = X.size(0)
        ones = torch.ones((num_samples, 1), device=X.device)
        xp = torch.cat((ones, X), dim=1).T

        d_t = d.T
        f_d = self.finv(d_t)
        derf = self.fderiv(f_d)

        F = torch.diag_embed(derf)
        H = torch.matmul(xp.unsqueeze(0), F)

        U, S, _ = torch.linalg.svd(H, full_matrices=False)
        f_d_vec = f_d.unsqueeze(-1)
        M = xp.unsqueeze(0) @ F @ (F @ f_d_vec)

        self.m = M.squeeze(-1)
        self.u = U
        self.s = S

    def forward(self, X: Tensor) -> Tensor:
        if not self.w:
            logger.warning("[ROLANN] No weights found")
            return torch.zeros((X.size(0), self.num_classes), device=X.device)

        num_samples = X.size(0)

        ones = torch.ones((num_samples, 1), device=X.device)
        xp = torch.cat((ones, X), dim=1).T

        W = torch.stack(list(self.w), dim=0)
        y_hat = self.f(torch.matmul(W, xp))

        return y_hat.T

    def _aggregate_parcial(self) -> None:
        for i in range(self.num_classes):
            m_k, u_k, s_k = self.m[i], self.u[i], self.s[i]

            if i >= len(self.mg):
                self.mg.append(nn.Parameter(m_k, requires_grad=False))
                self.ug.append(nn.Parameter(u_k, requires_grad=False))
                self.sg.append(nn.Parameter(s_k, requires_grad=False))

            else:
                m_g, u_g, s_g = self.mg[i], self.ug[i], self.sg[i]

                m_new = m_g + m_k
                us_g = u_g @ torch.diag(s_g)
                us_k = u_k @ torch.diag(s_k)

                concatenated = torch.cat((us_g, us_k), dim=1)
                u_new, s_new, _ = torch.linalg.svd(concatenated, full_matrices=False)

                self.mg[i] = nn.Parameter(m_new, requires_grad=False)
                self.ug[i] = nn.Parameter(u_new, requires_grad=False)
                self.sg[i] = nn.Parameter(s_new, requires_grad=False)

    def _calculate_weights(
        self,
    ) -> None:
        if not self.mg:
            return None

        new_w = nn.ParameterList()
        for i in range(self.num_classes):
            M = self.mg[i]
            U = self.ug[i]
            S = self.sg[i]
            diag = torch.diag(1 / (S * S))
            w_i = U @ (diag @ (U.T @ M))
            if w_i.dim() == 2 and w_i.shape[-1] == 1:
                w_i = w_i.squeeze(-1)
            new_w.append(nn.Parameter(w_i, requires_grad=False))
        self.w = new_w

    def aggregate_update(self, X: Tensor, d: Tensor) -> None:
        self.update_weights(X, d)  # The new M and US are calculated
        self._aggregate_parcial()  # New M and US added to old (global) ones
