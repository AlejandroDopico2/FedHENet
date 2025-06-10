# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 10:11:21 2024

@author: Oscar Fontenla & Alejandro Dopico
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torch.nn as nn


# Libreria para cifrado homomorfico
import tenseal as ts

class ROLANN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        lamb: float = 0.01,
        encrypted: bool = False, # Añadimos variable para cifrado
        context: ts.Context | None = None,
    ):
        super(ROLANN, self).__init__()

        self.num_classes = num_classes
        self.lamb = lamb  # Regularization hyperparameter

        self.f = torch.sigmoid
        self.finv = lambda x: torch.log(x / (1 - x))
        self.fderiv = lambda x: x * (1 - x)

        self.w = []

        self.m = None
        self.u = None
        self.s = None

        self.mg = []
        self.ug = []
        self.sg = []


        self.encrypted = encrypted

        if self.encrypted:

            if context is not None:
                self.context = context
            else:                              
                raise ValueError(
                    "Se necesita de un contexto para trabajar en modo encriptado. "
                )



    def update_weights(self, X: Tensor, d: Tensor) -> Tensor:
        results = [self._update_weights(X, d[:, i]) for i in range(self.num_classes)]

        ml, ul, sl = zip(*results)

        if self.encrypted:
            self.m = list(ml)
        else:
            # en modo normal, apilamos como tensor
            self.m = torch.stack(ml, dim=0)

        self.u = torch.stack(ul, dim=0)
        self.s = torch.stack(sl, dim=0)

    def _update_weights(self, X: Tensor, d: Tensor) -> Tensor:
        X = X.T
        n = X.size(1)  # Number of data points (n)

        # The bias is included as the first input (first row)
        ones = torch.ones((1, n), device=X.device)

        xp = torch.cat((ones, X), dim=0)

        # Inverse of the neural function
        f_d = self.finv(d)

        # Derivative of the neural function
        derf = self.fderiv(f_d)


        # Diagonal matrix
        F = torch.diag(derf)

        H = torch.matmul(xp, F)

        U, S, _ = torch.linalg.svd(H, full_matrices=False)

        M = torch.matmul(xp, torch.matmul(F, torch.matmul(F, f_d)))

        # encriptar M 
        if (self.encrypted):

            m_plain = M.detach().cpu().numpy().tolist()
            M = ts.ckks_vector(self.context, m_plain)

        return M, U, S

    def forward(self, X: Tensor) -> Tensor:
        X = X.T
        n = X.size(1)

        n_outputs = len(self.w)

        # Neural Network Simulation
        ones = torch.ones((1, n), device=X.device)
        xp = torch.cat((ones, X), dim=0)

        y_hat = torch.empty((n_outputs, n), device=X.device)

        for i in range(n_outputs):

            w_tmp = self.w[i].permute(
                *torch.arange(self.w[i].ndim - 1, -1, -1)
            )  # Trasposing

            y_hat[i] = self.f(torch.matmul(w_tmp, xp))

        return torch.transpose(y_hat, 0, 1)

    def _aggregate_parcial(self) -> None:
        for i in range(self.num_classes):
            if i >= len(self.mg):
                # Initialization using the first element of the list
                M = self.m[i]
                U = self.u[i]
                S = self.s[i]

                self.mg.append(M)
                self.ug.append(U)
                self.sg.append(S)

            else:
                M = self.mg[i]
                m_k = self.m[i]
                s_k = self.s[i]
                u_k = self.u[i]

                US = torch.matmul(self.ug[i], torch.diag(self.sg[i]))

                # Aggregation of M and US
                M = M + m_k
                us_k = torch.matmul(u_k, torch.diag(s_k))
                concatenated = torch.cat((us_k, US), dim=1)
                U, S, _ = torch.linalg.svd(concatenated, full_matrices=False)

                self.mg[i] = M
                self.ug[i] = U
                self.sg[i] = S

    def _calculate_weights(
        self,
    ) -> None:
        
        if not self.mg or not self.ug or not self.sg:
            return None

        # solo en cliente: desencriptar mg y generar self.w una sola vez
        new_w = []
        for i in range(self.num_classes):
            M = self.mg[i]
            # si es CKKSVector, lo desencriptamos
            if hasattr(M, "decrypt"):
                M = torch.tensor(M.decrypt(), device=self.ug[i].device, dtype=torch.float32)
            U = self.ug[i]
            S = self.sg[i]
            diag = torch.diag(1/(S*S + self.lamb))
            w_i = U @ (diag @ (U.T @ M))
            new_w.append(w_i)
        self.w = new_w

    def aggregate_update(self, X: Tensor, d: Tensor) -> None:
        self.update_weights(X, d)  # The new M and US are calculated
        self._aggregate_parcial()  # New M and US added to old (global) ones
        

