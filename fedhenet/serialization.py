"""Utilities for serializing tensors for MQTT transport.

All payloads produced here are JSON-safe (i.e. strings or dicts containing
only basic types) so they can be wrapped by the client/coordinator without
duplicating pickle/base64/CKKS logic across algorithms.
"""

from __future__ import annotations

import base64
import pickle
import zlib
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence

import numpy as np
import torch
import tenseal as ts


def _ensure_ctx(ctx: ts.Context | None) -> ts.Context:
    if ctx is None:
        raise ValueError("CKKS context is required for encrypted serialization.")
    return ctx


def _b64encode(data: bytes) -> str:
    return base64.b64encode(data).decode()


def _b64decode(data: str) -> bytes:
    return base64.b64decode(data.encode())


@dataclass(frozen=True)
class SerializationConfig:
    """Default configuration for tensor serialization."""

    encrypted: bool
    ctx: ts.Context | None = None
    compress: bool = False
    quantize_fp16: bool = False


class PayloadSerializer:
    """Serialize tensors with optional CKKS, compression and quantization."""

    def __init__(self, config: SerializationConfig):
        if config.encrypted:
            _ensure_ctx(config.ctx)
        self._config = config

    # ------------------------------------------------------------------#
    # Public helpers for named weight dictionaries
    # ------------------------------------------------------------------#
    def serialize_named_tensors(
        self,
        tensors: Mapping[str, torch.Tensor | Dict[str, Any] | ts.CKKSVector],
        *,
        include_shape: bool = True,
        overrides: Optional[SerializationConfig] = None,
    ) -> list[Dict[str, Any]]:
        """Serialize a dict of tensors into a payload list."""

        payload = []
        for name, tensor in tensors.items():
            payload.append(
                {
                    "name": name,
                    "weights": self._encode_value(
                        tensor, include_shape=include_shape, overrides=overrides
                    ),
                }
            )
        return payload

    def deserialize_named_tensors(
        self,
        payload: Iterable[Mapping[str, Any]],
        *,
        global_update: bool,
        include_shape: bool = True,
        return_cipher_wrapper: bool = True,
        overrides: Optional[SerializationConfig] = None,
    ) -> Dict[str, Any]:
        """Inverse operation for :meth:`serialize_named_tensors`."""

        weights: Dict[str, Any] = {}
        for entry in payload:
            encoded = entry["weights"]
            weights[entry["name"]] = self._decode_value(
                encoded,
                global_update=global_update,
                include_shape=include_shape,
                return_cipher_wrapper=return_cipher_wrapper,
                overrides=overrides,
            )
        return weights

    # ------------------------------------------------------------------#
    # Public helpers for arbitrary lists (FedHENet mg/us payloads)
    # ------------------------------------------------------------------#
    def serialize_sequence(
        self,
        values: Sequence[torch.Tensor | Dict[str, Any] | ts.CKKSVector],
        *,
        include_shape: bool = True,
        overrides: Optional[SerializationConfig] = None,
    ) -> list[Any]:
        return [
            self._encode_value(value, include_shape=include_shape, overrides=overrides)
            for value in values
        ]

    def deserialize_sequence(
        self,
        payload: Sequence[Any],
        *,
        global_update: bool,
        include_shape: bool = True,
        return_cipher_wrapper: bool = False,
        overrides: Optional[SerializationConfig] = None,
    ) -> list[Any]:
        return [
            self._decode_value(
                encoded,
                global_update=global_update,
                include_shape=include_shape,
                return_cipher_wrapper=return_cipher_wrapper,
                overrides=overrides,
            )
            for encoded in payload
        ]

    # ------------------------------------------------------------------#
    # Core encoding/decoding logic
    # ------------------------------------------------------------------#
    def _effective_config(
        self, overrides: Optional[SerializationConfig]
    ) -> SerializationConfig:
        if overrides is None:
            return self._config
        # Fill missing values from base config
        return SerializationConfig(
            encrypted=overrides.encrypted,
            ctx=overrides.ctx or self._config.ctx,
            compress=overrides.compress,
            quantize_fp16=overrides.quantize_fp16,
        )

    def _encode_value(
        self,
        value: torch.Tensor | Dict[str, Any] | ts.CKKSVector,
        *,
        include_shape: bool,
        overrides: Optional[SerializationConfig],
    ) -> Any:
        cfg = self._effective_config(overrides)

        if cfg.encrypted:
            return self._encode_encrypted(value, include_shape=include_shape, cfg=cfg)
        return self._encode_plain(value, cfg=cfg)

    def _encode_plain(
        self,
        value: torch.Tensor | Dict[str, Any],
        *,
        cfg: SerializationConfig,
    ) -> str:
        if isinstance(value, dict) and "data" in value:
            return value["data"]
        if not isinstance(value, torch.Tensor):
            raise TypeError("Plain serialization expects torch.Tensor values.")

        array = self._tensor_to_numpy(value, quantize_fp16=cfg.quantize_fp16)
        data = pickle.dumps(array, protocol=pickle.HIGHEST_PROTOCOL)
        if cfg.compress:
            data = zlib.compress(data)
        return _b64encode(data)

    def _encode_encrypted(
        self,
        value: torch.Tensor | Dict[str, Any] | ts.CKKSVector,
        *,
        include_shape: bool,
        cfg: SerializationConfig,
    ) -> Dict[str, Any]:
        ctx = _ensure_ctx(cfg.ctx)

        if isinstance(value, dict) and "cipher" in value:
            cipher = value["cipher"]
            shape = value.get("shape")
        elif isinstance(value, ts.CKKSVector):
            cipher = value
            shape = None
        else:
            if not isinstance(value, torch.Tensor):
                raise TypeError("Encrypted serialization expects torch.Tensor inputs.")
            array = self._tensor_to_numpy(value, quantize_fp16=cfg.quantize_fp16)
            shape = list(array.shape) if include_shape else None
            cipher = ts.ckks_vector(ctx, array.flatten().tolist())

        payload: Dict[str, Any] = {"data": _b64encode(cipher.serialize())}
        if include_shape and shape is not None:
            payload["shape"] = shape
        return payload

    def _decode_value(
        self,
        encoded: Any,
        *,
        global_update: bool,
        include_shape: bool,
        return_cipher_wrapper: bool,
        overrides: Optional[SerializationConfig],
    ) -> Any:
        cfg = self._effective_config(overrides)

        if cfg.encrypted:
            return self._decode_encrypted(
                encoded,
                global_update=global_update,
                include_shape=include_shape,
                return_cipher_wrapper=return_cipher_wrapper,
                cfg=cfg,
            )
        return self._decode_plain(encoded, cfg=cfg)

    def _decode_plain(self, encoded: Any, *, cfg: SerializationConfig) -> torch.Tensor:
        if not isinstance(encoded, str):
            raise TypeError("Plain payloads must be base64 strings.")
        data = _b64decode(encoded)
        if cfg.compress:
            data = zlib.decompress(data)
        array: np.ndarray = pickle.loads(data)
        tensor = torch.from_numpy(array).clone().detach()
        if tensor.dtype == torch.float16:
            tensor = tensor.to(torch.float32)
        return tensor

    def _decode_encrypted(
        self,
        encoded: Any,
        *,
        global_update: bool,
        include_shape: bool,
        return_cipher_wrapper: bool,
        cfg: SerializationConfig,
    ) -> Any:
        if not isinstance(encoded, Mapping) or "data" not in encoded:
            raise TypeError("Encrypted payload must contain serialized data.")
        ctx = _ensure_ctx(cfg.ctx)
        cipher = ts.ckks_vector_from(ctx, _b64decode(encoded["data"]))

        if global_update:
            flat = cipher.decrypt()
            tensor = torch.tensor(flat, dtype=torch.float32)
            if include_shape and "shape" in encoded:
                tensor = tensor.reshape(encoded["shape"])
            return tensor

        if return_cipher_wrapper:
            shape = encoded.get("shape") if include_shape else None
            return {"cipher": cipher, "shape": shape}
        return cipher

    @staticmethod
    def _tensor_to_numpy(
        tensor: torch.Tensor, *, quantize_fp16: bool
    ) -> np.ndarray:
        array = tensor.detach().cpu().numpy()
        if quantize_fp16:
            if np.isfinite(array).all():
                array = array.astype(np.float16)
        return array

