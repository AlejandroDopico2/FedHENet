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

class ROLANN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        encrypted: bool = False,
        context: ts.Context | None = None,
    ):
        super(ROLANN, self).__init__()

        self.num_classes = num_classes

        self.f = torch.sigmoid
        self.finv = lambda x: torch.log(x / (1 - x))
        self.fderiv = lambda x: x * (1 - x)

        self.w: List[Tensor] = []

        self.m: Optional[List[Tensor]] = None
        self.u: Optional[List[Tensor]] = None
        self.s: Optional[List[Tensor]] = None
        self.m_original_shapes: Optional[List[tuple]] = None

        self.mg: List[Tensor] = []
        self.ug: List[Tensor] = []
        self.sg: List[Tensor] = []

        self.encrypted: bool = encrypted
        self.context: Optional[ts.Context] = None

        if self.encrypted:
            if context is None:
                raise ValueError("A context is required to work in encrypted mode. ")

            self.context = context

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

        if self.encrypted:
            # Split M into per-class components and encrypt each separately
            m_list = []
            m_original_shapes = []
            for i in range(self.num_classes):
                m_i = M[i]  # Extract the i-th class component
                m_flat = m_i.flatten()
                m_plain = m_flat.detach().cpu().numpy().tolist()
                m_encrypted = ts.ckks_vector(self.context, m_plain)
                m_list.append(m_encrypted)
                m_original_shapes.append(m_i.shape)
            
            self.m = m_list
            self.m_original_shapes = m_original_shapes
        else:
            self.m = M.squeeze(-1)

        self.u = U
        self.s = S

    def forward(self, X: Tensor) -> Tensor:

        if not self.w:
            return torch.zeros((X.size(0), self.num_classes), device=X.device)

        num_samples = X.size(0)

        ones = torch.ones((num_samples, 1), device=X.device)
        xp = torch.cat((ones, X), dim=1).T

        W = torch.stack(self.w, dim=0)
        y_hat = self.f(torch.matmul(W, xp))

        return y_hat.T

    def _aggregate_parcial(self) -> None:
        for i in range(self.num_classes):
            m_k, u_k, s_k = self.m[i], self.u[i], self.s[i]

            if i >= len(self.mg):
                self.mg.append(m_k)
                self.ug.append(u_k)
                self.sg.append(s_k)

            else:
                m_g, u_g, s_g = self.mg[i], self.ug[i], self.sg[i]

                m_new = m_g + m_k
                us_g = u_g @ torch.diag(s_g)
                us_k = u_k @ torch.diag(s_k)

                concatenated = torch.cat((us_g, us_k), dim=1)
                u_new, s_new, _ = torch.linalg.svd(concatenated, full_matrices=False)

                self.mg[i] = m_new
                self.ug[i] = u_new
                self.sg[i] = s_new

    def _calculate_weights(
        self,
    ) -> None:
        if not self.mg:
            return None

        # only on client: decrypt mg and generate self.w only once
        new_w = []
        for i in range(self.num_classes):
            M = self.mg[i]
            # if it is CKKSVector, decrypt it
            if self.encrypted:
                M_flat = torch.tensor(
                    M.decrypt(), device=self.ug[i].device, dtype=torch.float32
                )
                # Reshape back to original dimensions for this class
                if hasattr(self, 'm_original_shapes') and self.m_original_shapes is not None and i < len(self.m_original_shapes):
                    M = M_flat.reshape(self.m_original_shapes[i])
                else:
                    # Fallback: assume it should be a column vector
                    M = M_flat.unsqueeze(-1)
            U = self.ug[i]
            S = self.sg[i]
            diag = torch.diag(1 / (S * S))
            w_i = U @ (diag @ (U.T @ M))
            # Store as 1D vector of length features+1
            if w_i.dim() == 2 and w_i.shape[-1] == 1:
                w_i = w_i.squeeze(-1)
            new_w.append(w_i)
        self.w = new_w

    def aggregate_update(self, X: Tensor, d: Tensor) -> None:
        self.update_weights(X, d)  # The new M and US are calculated
        self._aggregate_parcial()  # New M and US added to old (global) ones
