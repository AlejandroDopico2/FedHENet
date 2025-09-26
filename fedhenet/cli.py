import argparse
import time
from typing import List
import os
import logging

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from loguru import logger

from .config import load_config
from .datasets import prepare_splits, load_dataset
from .federated.client import Client
from .federated.coordinator import Coordinator
from .encrypted import create_context, serialize_context, deserialize_context
from .metrics import MetricsRecorder

os.environ['CODECARBON_LOG_LEVEL'] = 'WARNING'
logging.getLogger("codecarbon").setLevel(logging.WARNING)

logger.remove()
logger.add(
    lambda msg: print(msg, end=""),
    colorize=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level}</level> | "
    "<cyan>{function}</cyan>:"
    "<cyan>{line}</cyan> - <level>{message}</level>",
)

def run_experiment(config_path: str) -> None:

    logger.info(f"Loading config from {config_path}")
    cfg = load_config(config_path)

    # Optional external loggers
    wandb_run = None
    codecarbon_tracker = None
    if getattr(cfg, "logging", None) and cfg.logging.enable_wandb:
        try:
            import wandb  # type: ignore
            run_ts = time.strftime("%Y%m%d_%H%M")
            exp_name = (
                f"{cfg.dataset.name}-"
                f"{cfg.dataset.split}{'-alpha' if cfg.dataset.split == 'dirichlet' else ''}-"
                f"nc{cfg.dataset.num_clients}-"
                f"enc{int(bool(cfg.communication.encrypted))}-"
                f"{run_ts}"
            )
            exp_cfg = {
                "dataset.name": cfg.dataset.name,
                "dataset.split": cfg.dataset.split,
                "dataset.alpha": cfg.dataset.alpha if cfg.dataset.split == "dirichlet" else None,
                "dataset.num_clients": cfg.dataset.num_clients,
                "extractor.type": cfg.extractor.type,
                "num_classes": cfg.coordinator.num_classes,
                "coordinator.device": cfg.coordinator.device,
                "client.device": cfg.client.device,
                "client.batch_size": cfg.client.batch_size,
                "communication.encrypted": cfg.communication.encrypted,
                "communication.broker": cfg.communication.broker,
                "communication.port": cfg.communication.port,
            }
            wandb_run = wandb.init(
                project=cfg.logging.wandb_project or "fedhenet",
                name=exp_name,
                tags=cfg.logging.wandb_tags,
                config=exp_cfg,
                reinit=True,
            )
        except Exception as e:
            logger.warning(f"W&B disabled due to import/init error: {e}")

    if getattr(cfg, "logging", None) and cfg.logging.enable_codecarbon:
        try:
            from codecarbon import EmissionsTracker  # type: ignore
            codecarbon_tracker = EmissionsTracker(log_level="error")
            codecarbon_tracker.start()
        except Exception as e:
            logger.warning(f"CodeCarbon disabled due to import/init error: {e}")

    # Metrics recorder
    metrics = MetricsRecorder.instance()
    metrics.start()

    # Dataset splits per client
    logger.info(
        f"Preparing dataset: name={cfg.dataset.name}, split={cfg.dataset.split}, clients={cfg.dataset.num_clients}, subsample={cfg.dataset.subsample_fraction}"
    )
    client_datasets = prepare_splits(
        name=cfg.dataset.name,
        root=cfg.dataset.root,
        num_clients=cfg.dataset.num_clients,
        split=cfg.dataset.split,
        alpha=cfg.dataset.alpha,
        train=cfg.dataset.train,
        subsample_fraction=cfg.dataset.subsample_fraction,
    )
    logger.info("Datasets prepared for clients: " + ", ".join(str(len(d)) for d in client_datasets))

    # Devices with fallback if CUDA not available
    coord_device = cfg.coordinator.device
    client_device = cfg.client.device
    if client_device.lower() == "cuda" and not torch.cuda.is_available():
        logger.info("CUDA not available; falling back client device to CPU")
        client_device = "cpu"

    # Encryption contexts: build one master context and split into public/secret
    coord_ctx = None
    client_ctx = None
    if cfg.communication.encrypted:
        logger.info("Creating TenSEAL CKKS contexts")
        master_ctx = create_context()
        ctx_secret = serialize_context(master_ctx, secret_key=True)
        ctx_public = serialize_context(master_ctx, secret_key=False)
        client_ctx = deserialize_context(ctx_secret)
        coord_ctx = deserialize_context(ctx_public)

    logger.info("Starting coordinator")
    coord = Coordinator(
        num_classes=cfg.coordinator.num_classes,
        device=coord_device,
        num_clients=cfg.dataset.num_clients,
        encrypted=cfg.communication.encrypted,
        ctx=coord_ctx,
        broker=cfg.communication.broker,
        port=cfg.communication.port,
    )

    logger.info(f"Initializing clients")
    clients: List[Client] = []
    for i, ds in enumerate(client_datasets):
        client = Client(
            num_classes=cfg.coordinator.num_classes,
            dataset=ds,
            device=client_device,
            client_id=i,
            encrypted=cfg.communication.encrypted,
            ctx=client_ctx,
            broker=cfg.communication.broker,
            port=cfg.communication.port,
        )
        clients.append(client)

    # Local training and send updates
    logger.info("Starting local training across clients")

    for c in tqdm(clients, desc="Clients", leave=True):
        c.training()
        c.aggregate_parcial()
    logger.info("Local training done; updates published")

    # Wait for global model broadcast (with timeout per client)
    logger.info("Waiting for global model broadcast")
    ready_count = 0
    for c in clients:
        if hasattr(c, 'wait_for_global_model') and c.wait_for_global_model(timeout_s=30.0):
            ready_count += 1
    if ready_count < len(clients):
        logger.warning(f"Only {ready_count}/{len(clients)} clients received the global model in time")

    # Evaluate on a single held-out test dataset (no partitioning)
    test_ds = load_dataset(name=cfg.dataset.name, root=cfg.dataset.root, train=False, download=True)
    loader = DataLoader(test_ds, batch_size=32)

    acc = clients[0].evaluate(loader)
    logger.info(f"Client 0 test acc = {acc:.2f}")

    # Stop timers
    metrics.end()

    # Log to external systems
    snapshot = metrics.snapshot()
    logger.info(
        f"Run stats: time={snapshot['elapsed_seconds']:.2f}s, pub={snapshot['published_bytes']} bytes, recv={snapshot['received_bytes']} bytes"
    )
    if wandb_run is not None:
        try:
            import wandb  # type: ignore
            wandb.log({
                "accuracy": acc,
                "elapsed_seconds": snapshot["elapsed_seconds"],
                "mqtt_published_bytes": snapshot["published_bytes"],
                "mqtt_received_bytes": snapshot["received_bytes"],
            })
        except Exception as e:
            logger.warning(f"W&B log error: {e}")

    if codecarbon_tracker is not None:
        try:
            emissions = codecarbon_tracker.stop()
            emissions = codecarbon_tracker.final_emissions_data
            if emissions is not None and wandb_run is not None:
                import wandb  # type: ignore
                wandb.log({
                    "codecarbon_emissions_kg": emissions.emissions,
                    "codecarbon_energy_kwh": emissions.energy_consumed
                })
            if emissions is not None:
                logger.info(
                    f"Energy: {emissions.energy_consumed:.6f} kWh, Emissions: {emissions.emissions:.6f} kgCO2"
                )
        except Exception as e:
            logger.warning(f"CodeCarbon stop error: {e}")

    if wandb_run is not None:
        try:
            wandb_run.finish()  # type: ignore
        except Exception:
            pass

    # Graceful shutdown of MQTT clients to avoid core dumps
    try:
        for c in clients:
            if hasattr(c, 'shutdown'):
                c.shutdown()
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(prog="fedhenet", description="FedHeNet CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    sim = sub.add_parser("simulate", help="Run single-machine simulation from a TOML config")
    sim.add_argument("--config", required=True, help="Path to TOML config file")

    args = parser.parse_args()
    if args.command == "simulate":
        run_experiment(args.config)


simulate_from_config = run_experiment

__all__ = ["main", "run_experiment", "simulate_from_config"]


