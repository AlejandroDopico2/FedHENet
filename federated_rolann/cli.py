import argparse
import time
from typing import List

import torch

from .config import load_config
from .datasets import prepare_splits
from .encrypted import create_context, serialize_context, deserialize_context
from .federated.client import Client
from .federated.coordinator import Coordinator


def simulate_from_config(config_path: str) -> None:
    cfg = load_config(config_path)

    # Dataset splits per client
    client_datasets = prepare_splits(
        name=cfg.dataset.name,
        root=cfg.dataset.root,
        num_clients=cfg.dataset.num_clients,
        split=cfg.dataset.split,
        alpha=cfg.dataset.alpha,
        train=cfg.dataset.train,
        subsample_fraction=cfg.dataset.subsample_fraction,
    )

    # Encryption contexts (encryption-first default if enabled)
    coord_ctx = None
    client_ctx = None
    if cfg.coordinator.encrypted or cfg.client.encrypted:
        # If paths provided, load; else generate fresh
        if cfg.coordinator.context_public_path and cfg.client.context_secret_path:
            with open(cfg.client.context_secret_path, "rb") as f:
                client_ctx = deserialize_context(f.read())
            with open(cfg.coordinator.context_public_path, "rb") as f:
                coord_ctx = deserialize_context(f.read())
        else:
            master = create_context()
            ctx_secret = serialize_context(master, secret_key=True)
            ctx_public = serialize_context(master, secret_key=False)
            client_ctx = deserialize_context(ctx_secret)
            coord_ctx = deserialize_context(ctx_public)

    # Coordinator
    coord = Coordinator(
        num_classes=cfg.coordinator.num_classes,
        device=cfg.coordinator.device,
        num_clients=cfg.coordinator.num_clients,
        encrypted=cfg.coordinator.encrypted,
        ctx=coord_ctx,
        broker=cfg.coordinator.broker,
        port=cfg.coordinator.port,
    )

    # Clients
    clients: List[Client] = []
    for i, ds in enumerate(client_datasets):
        client = Client(
            num_classes=cfg.coordinator.num_classes,
            dataset=ds,
            device=cfg.client.device,
            client_id=i,
            encrypted=cfg.client.encrypted,
            ctx=client_ctx,
            broker=cfg.client.broker,
            port=cfg.client.port,
        )
        clients.append(client)

    # Local training and send updates
    for c in clients:
        c.training()
        c.aggregate_parcial()

    # Allow time for broadcast
    time.sleep(2)

    # Evaluate on union of client datasets (quick check)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.ConcatDataset(client_datasets), batch_size=32
    )
    for i, c in enumerate(clients):
        acc = c.evaluate(loader)
        print(f"Client {i}: acc = {acc:.2f}")


def main():
    parser = argparse.ArgumentParser(prog="fedhenet", description="FedHeNet CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    sim = sub.add_parser("simulate", help="Run single-machine simulation from a TOML config")
    sim.add_argument("--config", required=True, help="Path to TOML config file")

    args = parser.parse_args()
    if args.command == "simulate":
        simulate_from_config(args.config)


__all__ = ["main", "simulate_from_config"]


