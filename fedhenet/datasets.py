import math
import random
from typing import List, Optional

import torch
from torch.utils.data import Subset
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, VisionDataset
from torchvision import transforms as T

from loguru import logger


def load_dataset(
    name: str,
    root: str,
    train: bool = True,
    download: bool = True,
    transform=None,
) -> VisionDataset:
    """
    Minimal dataset loader for CIFAR-10/100 and MNIST.
    """
    if transform is None:
        logger.info(f"Loading dataset transforms for {name}")
        lname = name.lower()
        if lname == "mnist":
            transform = T.Compose(
                [
                    T.Grayscale(num_output_channels=3),
                    T.Resize(224),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        elif lname == "cifar10":
            transform = T.Compose(
                [
                    T.Resize((224, 224)),
                    T.ToTensor(),
                    T.Normalize(
                        mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
                    ),
                ]
            )
        elif lname == "cifar100":
            transform = T.Compose(
                [
                    T.Resize((224, 224)),
                    T.ToTensor(),
                    T.Normalize(
                        mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]
                    ),
                ]
            )

    lname = name.lower()
    if lname == "cifar10":
        return CIFAR10(root=root, train=train, download=download, transform=transform)
    if lname == "cifar100":
        return CIFAR100(root=root, train=train, download=download, transform=transform)
    if lname == "mnist":
        return MNIST(root=root, train=train, download=download, transform=transform)
    raise ValueError(
        f"Unsupported dataset: {name}. Supported: cifar10, cifar100, mnist"
    )


def _get_targets(dataset: VisionDataset) -> torch.Tensor:
    """Return labels from a torchvision dataset as a 1D torch tensor."""
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


def subsample_dataset(
    dataset: VisionDataset, fraction: float, seed: int = 42
) -> Subset:
    """Return a random subset with the given fraction of items."""
    if not (0.0 < fraction <= 1.0):
        raise ValueError("fraction must be in (0, 1]")
    rng = random.Random(seed)
    n = len(dataset)
    k = max(1, int(math.ceil(n * fraction)))
    indices = list(range(n))
    rng.shuffle(indices)
    return Subset(dataset, indices[:k])


def split_iid(dataset: VisionDataset, num_clients: int, seed: int = 42) -> List[Subset]:
    """Return IID shards split uniformly into num_clients subsets."""
    if num_clients <= 0:
        raise ValueError("num_clients must be > 0")
    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    shards = [indices[i::num_clients] for i in range(num_clients)]
    return [Subset(dataset, shard) for shard in shards]  # type: ignore


def split_dirichlet(
    dataset: VisionDataset,
    num_clients: int,
    alpha: float = 0.5,
    seed: int = 42,
) -> List[Subset[VisionDataset]]:
    """Return non-IID Dirichlet shards across clients."""
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


def split_single_class(
    dataset: VisionDataset,
    num_clients: int,
    alpha: float = 0.5,
    seed: int = 42,
) -> List[Subset[VisionDataset]]:
    """Assign each client data from exactly one class ensuring all classes appear.

    Requires num_clients >= num_classes. Clients are distributed across classes
    as evenly as possible.
    """
    if num_clients <= 0:
        raise ValueError("num_clients must be > 0")
    if alpha <= 0:
        raise ValueError("alpha must be > 0")

    targets = _get_targets(dataset)
    num_classes = int(targets.max().item() + 1)

    if num_clients < num_classes:
        raise ValueError(
            f"num_clients ({num_clients}) must be >= num_classes ({num_classes}) for 'single_class' split"
        )

    # Build per-class shuffled indices
    class_indices: List[List[int]] = []
    for c in range(num_classes):
        idx = torch.nonzero(targets == c, as_tuple=False).view(-1).tolist()
        random.Random(seed + c).shuffle(idx)
        if len(idx) == 0:
            raise ValueError(f"Class {c} has no samples in the dataset")
        class_indices.append(idx)

    # Determine how many clients are assigned to each class (at least one per class)
    base = num_clients // num_classes
    remainder = num_clients % num_classes
    clients_per_class = [base + (1 if i < remainder else 0) for i in range(num_classes)]

    # Prepare client buckets
    client_indices: List[List[int]] = [[] for _ in range(num_clients)]
    client_class: List[int] = [-1 for _ in range(num_clients)]  # track assigned class

    # Map clients to classes in order and slice per-class indices accordingly
    client_id = 0
    for cls, k in enumerate(clients_per_class):
        if k <= 0:
            continue
        idx_c = class_indices[cls]

        # Allocate this class's samples to its k clients using Dirichlet(alpha)
        n = len(idx_c)
        if n < k:
            raise ValueError(
                f"Not enough samples in class {cls} ({n}) to assign at least 1 to each of {k} clients"
            )

        # Guarantee at least 1 sample per client, distribute the remaining via Dirichlet
        remaining = n - k
        if remaining == 0:
            counts = [1 for _ in range(k)]
        else:
            concentration = torch.full((k,), float(alpha))
            props = torch.distributions.Dirichlet(concentration).sample((1,)).squeeze(0)
            props = (props / props.sum()).tolist()
            extra = [int(round(p * remaining)) for p in props]
            # Fix rounding drift to exactly match 'remaining'
            drift = remaining - sum(extra)
            for j in range(abs(drift)):
                extra[j % k] += 1 if drift > 0 else -1
            counts = [1 + e for e in extra]

        start = 0
        for j in range(k):
            cnt = counts[j]
            if cnt <= 0:
                continue
            part = idx_c[start : start + cnt]
            client_indices[client_id].extend(part)
            client_class[client_id] = cls
            client_id += 1
            start += cnt

    # Shuffle per-client shard for randomness
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
) -> List[Subset[VisionDataset]]:
    """Load dataset and return client subsets per requested split."""
    ds = load_dataset(
        name=name, root=root, train=train, download=True, transform=transform
    )
    if subsample_fraction is not None and subsample_fraction < 1.0:
        ds = subsample_dataset(ds, subsample_fraction, seed=seed)

    split_lc = split.lower()
    if split_lc == "iid":
        return split_iid(ds, num_clients=num_clients, seed=seed)
    if split_lc in ("dirichlet", "dir"):
        return split_dirichlet(ds, num_clients=num_clients, alpha=alpha, seed=seed)
    if split_lc in ("single", "single_class", "one_class"):
        return split_single_class(ds, num_clients=num_clients, alpha=alpha, seed=seed)
    raise ValueError(
        f"Unsupported split: {split}. Use 'iid', 'dirichlet', or 'single_class'."
    )


__all__ = [
    "load_dataset",
    "subsample_dataset",
    "split_iid",
    "split_dirichlet",
    "split_single_class",
    "prepare_splits",
]
