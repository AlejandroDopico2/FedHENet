import argparse
import time
from typing import List
import os
import logging
import numpy as np
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

os.environ["CODECARBON_LOG_LEVEL"] = "WARNING"
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


class ExperimentRunner:
    def __init__(self, config_path: str):
        """Initialize the experiment runner with configuration and setup logging, trackers, and metrics."""
        logger.info(f"Loading config from {config_path}")
        self.cfg = load_config(config_path)

        # TODO: Later config overrides could be merged here (e.g., CLI args)

        # Optional external loggers
        self.wandb_run = None
        self.codecarbon_tracker = None
        if getattr(self.cfg, "logging", None) and self.cfg.logging.enable_wandb:
            try:
                import wandb  # type: ignore

                run_ts = time.strftime("%Y%m%d_%H%M")
                exp_name = (
                    f"fedhenet-{self.cfg.dataset.name}-"
                    f"{self.cfg.dataset.split}{'-alpha' if self.cfg.dataset.split == 'dirichlet' else ''}-"
                    f"nc{self.cfg.dataset.num_clients}-"
                    f"enc{int(bool(self.cfg.communication.encrypted))}-"
                    f"{run_ts}"
                )
                exp_cfg = {
                    "dataset.name": self.cfg.dataset.name,
                    "dataset.split": self.cfg.dataset.split,
                    "dataset.alpha": (
                        self.cfg.dataset.alpha
                        if self.cfg.dataset.split == "dirichlet"
                        else None
                    ),
                    "dataset.num_clients": self.cfg.dataset.num_clients,
                    "extractor.type": self.cfg.extractor.type,
                    "num_classes": self.cfg.coordinator.num_classes,
                    "coordinator.device": self.cfg.coordinator.device,
                    "client.device": self.cfg.client.device,
                    "client.batch_size": self.cfg.client.batch_size,
                    "communication.encrypted": self.cfg.communication.encrypted,
                    "communication.broker": self.cfg.communication.broker,
                    "communication.port": self.cfg.communication.port,
                }
                self.wandb_run = wandb.init(
                    project=self.cfg.logging.wandb_project or "fedhenet",
                    name=exp_name,
                    tags=self.cfg.logging.wandb_tags,
                    config=exp_cfg,
                    reinit=True,
                )
            except Exception as e:
                logger.warning(f"W&B disabled due to import/init error: {e}")

        if getattr(self.cfg, "logging", None) and self.cfg.logging.enable_codecarbon:
            try:
                from codecarbon import EmissionsTracker  # type: ignore

                self.codecarbon_tracker = EmissionsTracker(log_level="error")
                self.codecarbon_tracker.start()
            except Exception as e:
                logger.warning(f"CodeCarbon disabled due to import/init error: {e}")

        # Metrics recorder
        self.metrics = MetricsRecorder.instance()

        # Initialize instance variables
        self.client_datasets = None
        self.coord = None
        self.clients = None

    def prepare_datasets(self):
        """Handle dataset splitting."""
        # Dataset splits per client
        logger.info(
            f"Preparing dataset: name={self.cfg.dataset.name}, split={self.cfg.dataset.split}, clients={self.cfg.dataset.num_clients}, subsample={self.cfg.dataset.subsample_fraction}"
        )
        self.client_datasets = prepare_splits(
            name=self.cfg.dataset.name,
            root=self.cfg.dataset.root,
            num_clients=self.cfg.dataset.num_clients,
            split=self.cfg.dataset.split,
            alpha=self.cfg.dataset.alpha,
            subsample_fraction=self.cfg.dataset.subsample_fraction,
        )
        logger.info(
            "Datasets prepared for clients: "
            + ", ".join(str(len(d)) for d in self.client_datasets)
        )

    def init_clients(self):
        """Initialize coordinator and clients."""
        # Devices with fallback if CUDA not available
        coord_device = self.cfg.coordinator.device
        client_device = self.cfg.client.device
        if client_device.lower() == "cuda" and not torch.cuda.is_available():
            logger.info("CUDA not available; falling back client device to CPU")
            client_device = "cpu"

        # Encryption contexts: build one master context and split into public/secret
        coord_ctx = None
        client_ctx = None
        if self.cfg.communication.encrypted:
            logger.info("Creating TenSEAL CKKS contexts")
            master_ctx = create_context()
            ctx_secret = serialize_context(master_ctx, secret_key=True)
            ctx_public = serialize_context(master_ctx, secret_key=False)
            client_ctx = deserialize_context(ctx_secret)
            coord_ctx = deserialize_context(ctx_public)

        logger.info("Starting coordinator")
        self.coord = Coordinator(
            num_classes=self.cfg.coordinator.num_classes,
            device=coord_device,
            num_clients=self.cfg.dataset.num_clients,
            encrypted=self.cfg.communication.encrypted,
            ctx=coord_ctx,
            broker=self.cfg.communication.broker,
            port=self.cfg.communication.port,
        )

        logger.info("Initializing clients")
        self.clients = []
        for i, ds in enumerate(self.client_datasets):
            client = Client(
                num_classes=self.cfg.coordinator.num_classes,
                dataset=ds,
                device=client_device,
                client_id=i,
                encrypted=self.cfg.communication.encrypted,
                ctx=client_ctx,
                broker=self.cfg.communication.broker,
                port=self.cfg.communication.port,
            )
            self.clients.append(client)

    def train_and_evaluate(self) -> List[float]:
        """Run local training, aggregation, and evaluation."""
        # Local training and send updates
        logger.info("Starting local training across clients")

        test_ds = load_dataset(
            name=self.cfg.dataset.name,
            root=self.cfg.dataset.root,
            train=False,
            download=True,
        )
        loader = DataLoader(test_ds, batch_size=32)

        for c in tqdm(self.clients, desc="Clients", leave=True):
            c.training()
            acc = c.evaluate(loader)
            c.aggregate_parcial()
        logger.info("Local training done; updates published")

        logger.info("Waiting for global model broadcast")
        ready_count = 0
        for c in self.clients:
            if hasattr(c, "wait_for_global_model") and c.wait_for_global_model(
                timeout_s=30.0
            ):
                ready_count += 1
        if ready_count < len(self.clients):
            logger.warning(
                f"Only {ready_count}/{len(self.clients)} clients received the global model in time"
            )

        # Evaluate on a single held-out test dataset (no partitioning)
        test_ds = load_dataset(
            name=self.cfg.dataset.name,
            root=self.cfg.dataset.root,
            train=False,
            download=True,
        )
        loader = DataLoader(test_ds, batch_size=32)

        accuracies = []
        for c in self.clients:
            acc = c.evaluate(loader)
            accuracies.append(acc)
        return accuracies

    def _safe_shutdown(self):
        """Safely shutdown coordinator and clients with error handling."""
        try:
            if self.coord is not None:
                self.coord.shutdown()
        except Exception:
            pass
        try:
            if self.clients is not None:
                for c in self.clients:
                    c.shutdown()
        except Exception:
            pass

    def finalize(self, accuracies: List[float]):
        """Shut down coordinator, clients, trackers, and log metrics."""
        self._safe_shutdown()

        self.metrics.end()

        # Log to external systems
        snapshot = self.metrics.snapshot()
        logger.info(
            f"Run stats: time={snapshot['elapsed_seconds']:.2f}s, pub={snapshot['published_bytes']} bytes, recv={snapshot['received_bytes']} bytes"
        )
        if self.wandb_run is not None:
            try:
                import wandb  # type: ignore

                mean_accuracy = np.mean(accuracies)
                std_accuracy = np.std(accuracies)

                wandb.log(
                    {
                        "accuracy": mean_accuracy,
                        "accuracy_std": std_accuracy,
                        "elapsed_seconds": snapshot["elapsed_seconds"],
                        "mqtt_published_bytes": snapshot["published_bytes"],
                        "mqtt_received_bytes": snapshot["received_bytes"],
                    }
                )
            except Exception as e:
                logger.warning(f"W&B log error: {e}")

        if self.codecarbon_tracker is not None:
            try:
                emissions = self.codecarbon_tracker.stop()
                emissions = self.codecarbon_tracker.final_emissions_data
                if emissions is not None and self.wandb_run is not None:
                    import wandb  # type: ignore

                    wandb.log(
                        {
                            "codecarbon_emissions_kg": emissions.emissions,
                            "codecarbon_energy_kwh": emissions.energy_consumed,
                        }
                    )
                if emissions is not None:
                    logger.info(
                        f"Energy: {emissions.energy_consumed:.6f} kWh, Emissions: {emissions.emissions:.6f} kgCO2"
                    )
            except Exception as e:
                logger.warning(f"CodeCarbon stop error: {e}")

        if self.wandb_run is not None:
            try:
                self.wandb_run.finish()  # type: ignore
            except Exception:
                pass

        self._safe_shutdown()

    def run(self):
        """Orchestrate the experiment steps in correct order."""
        try:
            self.prepare_datasets()
            self.init_clients()
            self.metrics.start()
            accuracies = self.train_and_evaluate()
            self.finalize(accuracies)
        except Exception as e:
            # Ensure cleanup even if training fails
            self._safe_shutdown()
            raise e
        finally:
            if self.metrics is not None:
                try:
                    self.metrics.end()
                except Exception:
                    pass
            if self.wandb_run is not None:
                try:
                    self.wandb_run.finish()
                except Exception:
                    pass
            if self.codecarbon_tracker is not None:
                try:
                    self.codecarbon_tracker.stop()
                except Exception:
                    pass
            raise e


def run_experiment(config_path: str) -> None:
    """Thin wrapper for backward compatibility."""
    runner = ExperimentRunner(config_path)
    runner.run()


def main():
    parser = argparse.ArgumentParser(prog="fedhenet", description="FedHeNet CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    sim = sub.add_parser(
        "simulate", help="Run single-machine simulation from a TOML config"
    )
    sim.add_argument("--config", required=True, help="Path to TOML config file")

    args = parser.parse_args()
    if args.command == "simulate":
        ExperimentRunner(args.config).run()


simulate_from_config = run_experiment

__all__ = ["main", "run_experiment", "simulate_from_config"]
