from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# Python 3.11+: stdlib tomllib. For 3.10, try tomli if available.
try:
    import tomllib  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    tomllib = None  # type: ignore
    try:
        import tomli as _tomli  # type: ignore
    except Exception:
        _tomli = None  # type: ignore


@dataclass
class DatasetConfig:
    name: str = "cifar10"  # cifar10|mnist|fake
    root: str = "./data"
    split: str = "iid"  # iid|dirichlet
    alpha: float = 0.5
    num_clients: int = 2
    subsample_fraction: Optional[float] = None
    train: bool = True


@dataclass
class ExtractorConfig:
    type: str = "resnet18"
    weights: str = "default"  # default|none


@dataclass
class CoordinatorConfig:
    num_classes: int = 10
    device: str = "cpu"
    num_clients: int = 2
    broker: str = "localhost"
    port: int = 1883
    encrypted: bool = False
    context_public_path: Optional[str] = None


@dataclass
class ClientConfig:
    device: str = "cpu"
    batch_size: int = 128
    encrypted: bool = False
    broker: str = "localhost"
    port: int = 1883
    context_secret_path: Optional[str] = None


@dataclass
class Config:
    dataset: DatasetConfig = DatasetConfig()
    extractor: ExtractorConfig = ExtractorConfig()
    coordinator: CoordinatorConfig = CoordinatorConfig()
    client: ClientConfig = ClientConfig()


def _parse_toml(path: str) -> dict:
    with open(path, "rb") as f:
        if tomllib is not None:
            return tomllib.load(f)  # type: ignore
        if '_tomli' in globals() and globals()['_tomli'] is not None:
            return globals()['_tomli'].load(f)  # type: ignore
        raise RuntimeError("No TOML parser available. Use Python 3.11+ or install tomli.")


def load_config(path: str) -> Config:
    """
    Load a TOML config file into Config dataclasses.
    Expected sections: [dataset], [extractor], [coordinator], [client].
    """
    data = _parse_toml(path)

    ds = data.get("dataset", {})
    ex = data.get("extractor", {})
    co = data.get("coordinator", {})
    cl = data.get("client", {})

    cfg = Config(
        dataset=DatasetConfig(**ds) if ds else DatasetConfig(),
        extractor=ExtractorConfig(**ex) if ex else ExtractorConfig(),
        coordinator=CoordinatorConfig(**co) if co else CoordinatorConfig(),
        client=ClientConfig(**cl) if cl else ClientConfig(),
    )
    return cfg


__all__ = [
    "DatasetConfig",
    "ExtractorConfig",
    "CoordinatorConfig",
    "ClientConfig",
    "Config",
    "load_config",
]


