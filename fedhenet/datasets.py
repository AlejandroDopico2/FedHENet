import math
import random
from typing import List, Sequence, Tuple, Optional

import torch
from torch.utils.data import Subset
from torchvision.datasets import CIFAR10, MNIST
from torchvision.transforms import ToTensor


def load_dataset(
    name: str,
    root: str,
    train: bool = True,
    download: bool = True,
    transform=None,
):
    """
    Minimal dataset loader for CIFAR-10 and MNIST.
    Returns a torchvision Dataset.
    """
    if transform is None:
        transform = ToTensor()

    lname = name.lower()
    if lname == "cifar10":
        return CIFAR10(root=root, train=train, download=download, transform=transform)
    if lname == "mnist":
        return MNIST(root=root, train=train, download=download, transform=transform)
    raise ValueError(f"Unsupported dataset: {name}. Supported: cifar10, mnist")


def _get_targets(dataset) -> torch.Tensor:
    """Extract labels from a torchvision dataset as a 1D torch tensor."""
    # Common attributes in torchvision datasets
    for attr in ("targets", "labels"):
        if hasattr(dataset, attr):
            data = getattr(dataset, attr)
            if isinstance(data, torch.Tensor):
                return data.view(-1).clone()
            return torch.tensor(list(data), dtype=torch.long)

    # Fallback: index the dataset (slower)
    labels = []
    for i in range(len(dataset)):
        _, y = dataset[i]
        if isinstance(y, torch.Tensor):
            y = int(y.item())
        labels.append(int(y))
    return torch.tensor(labels, dtype=torch.long)


def subsample_dataset(dataset, fraction: float, seed: int = 42) -> Subset:
    """
    Randomly subsample a fraction of the dataset. Returns a Subset.
    """
    if not (0.0 < fraction <= 1.0):
        raise ValueError("fraction must be in (0, 1]")
    rng = random.Random(seed)
    n = len(dataset)
    k = max(1, int(math.ceil(n * fraction)))
    indices = list(range(n))
    rng.shuffle(indices)
    return Subset(dataset, indices[:k])


def split_iid(dataset, num_clients: int, seed: int = 42) -> List[Subset]:
    """
    IID split: random uniform partition into num_clients roughly equal shards.
    Returns a list of Subset (one per client).
    """
    if num_clients <= 0:
        raise ValueError("num_clients must be > 0")
    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    shards = [indices[i::num_clients] for i in range(num_clients)]
    return [Subset(dataset, shard) for shard in shards]


def split_dirichlet(
    dataset,
    num_clients: int,
    alpha: float = 0.5,
    seed: int = 42,
) -> List[Subset]:
    """
    Non-IID Dirichlet split across clients. For each class, allocate samples to
    clients according to Dirichlet(alpha) proportions.
    Returns a list of Subset (one per client).
    """
    if num_clients <= 0:
        raise ValueError("num_clients must be > 0")
    if alpha <= 0:
        raise ValueError("alpha must be > 0")

    targets = _get_targets(dataset)
    num_classes = int(targets.max().item() + 1)

    # Build per-class index lists
    class_indices: List[List[int]] = []
    for c in range(num_classes):
        idx = torch.nonzero(targets == c, as_tuple=False).view(-1).tolist()
        random.Random(seed + c).shuffle(idx)
        class_indices.append(idx)

    # Initialize client index buckets
    client_indices: List[List[int]] = [[] for _ in range(num_clients)]

    # Torch Dirichlet for reproducibility without adding numpy dep
    concentration = torch.full((num_clients,), float(alpha))
    # Use global RNG; keep reproducibility via Python's random + fixed ordering above

    for c in range(num_classes):
        idx_c = class_indices[c]
        if not idx_c:
            continue
        # Sample proportions and convert to counts
        props = torch.distributions.Dirichlet(concentration).sample((1,)).squeeze(0)
        props = (props / props.sum()).tolist()
        counts = [int(round(p * len(idx_c))) for p in props]

        # Fix rounding drift to match total exactly
        drift = len(idx_c) - sum(counts)
        for i in range(abs(drift)):
            counts[i % num_clients] += 1 if drift > 0 else -1

        # Slice and assign
        start = 0
        for client_id, cnt in enumerate(counts):
            if cnt <= 0:
                continue
            client_indices[client_id].extend(idx_c[start : start + cnt])
            start += cnt

    # Shuffle each client shard for randomness
    for i in range(num_clients):
        random.Random(seed + 10 + i).shuffle(client_indices[i])

    return [Subset(dataset, idxs) for idxs in client_indices]


def prepare_splits(
    name: str,
    root: str,
    num_clients: int,
    split: str = "iid",
    alpha: float = 0.5,
    train: bool = True,
    subsample_fraction: Optional[float] = None,
    seed: int = 42,
    transform=None,
):
    """
    Convenience: load dataset and return client Subsets per requested split.
    """
    ds = load_dataset(name=name, root=root, train=train, download=True, transform=transform)
    if subsample_fraction is not None and subsample_fraction < 1.0:
        ds = subsample_dataset(ds, subsample_fraction, seed=seed)

    split_lc = split.lower()
    if split_lc == "iid":
        return split_iid(ds, num_clients=num_clients, seed=seed)
    if split_lc in ("dirichlet", "dir"):
        return split_dirichlet(ds, num_clients=num_clients, alpha=alpha, seed=seed)
    raise ValueError(f"Unsupported split: {split}. Use 'iid' or 'dirichlet'.")


__all__ = [
    "load_dataset",
    "subsample_dataset",
    "split_iid",
    "split_dirichlet",
    "prepare_splits",
]


