"""
Experiment runner module for FedHeNet.

This module contains the ExperimentRunner class that orchestrates federated learning experiments.
"""

import time
import random
from typing import List, Union
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from loguru import logger

from .datasets import prepare_splits, load_dataset
from .federated.client import Client
from .federated.coordinator import Coordinator
from .encrypted import create_context, serialize_context, deserialize_context
from .metrics import MetricsRecorder

try:
    from omegaconf import DictConfig
except ImportError:
    DictConfig = dict  # Fallback for when Hydra is not available


class ExperimentRunner:
    def __init__(self, config: Union[str, DictConfig]):
        """Initialize the experiment runner with configuration and setup logging, trackers, and metrics."""
        if isinstance(config, str):
            from .config import load_config
            logger.info(f"Loading config from {config}")
            self.cfg = load_config(config)
        else:
            logger.info("Using provided DictConfig")
            self.cfg = config

        # TODO: Later config overrides could be merged here (e.g., CLI args)

        # Set random seeds for reproducibility
        self._set_seeds()

        # Optional external loggers
        self.wandb_run = None
        self.codecarbon_tracker = None
        if getattr(self.cfg, "logging", None) and self.cfg.logging.enable_wandb:
            try:
                import wandb  # type: ignore

                run_ts = time.strftime("%Y%m%d_%H%M")
                exp_name = (
                    f"fedhenet-{self.cfg.dataset.name}-"
                    f"{self.cfg.dataset.split}{f'-alpha{self.cfg.dataset.alpha}' if self.cfg.dataset.split == 'dirichlet' else ''}-"
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
                    config=exp_cfg,
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

    def _set_seeds(self):
        """Set random seeds for reproducibility."""
        seed = getattr(self.cfg, 'seed', None)
        if seed is not None:
            logger.info(f"Setting random seed to {seed}")
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            # For deterministic behavior (slower but more reproducible)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            logger.info("No seed specified, using random initialization")

    def prepare_datasets(self):
        """Handle dataset splitting."""
        # Dataset splits per client
        logger.info(
            f"Preparing dataset: name={self.cfg.dataset.name}, split={self.cfg.dataset.split}, clients={self.cfg.dataset.num_clients}, subsample={self.cfg.dataset.subsample_fraction}"
        )
        seed = getattr(self.cfg, 'seed', None)
        self.client_datasets = prepare_splits(
            name=self.cfg.dataset.name,
            root=self.cfg.dataset.root,
            num_clients=self.cfg.dataset.num_clients,
            split=self.cfg.dataset.split,
            alpha=self.cfg.dataset.alpha,
            subsample_fraction=self.cfg.dataset.subsample_fraction,
            seed=seed,
        )
        logger.info(
            "Datasets prepared for clients: "
            + ", ".join(str(len(d)) for d in self.client_datasets)
        )

        self.test_ds = load_dataset(
            name=self.cfg.dataset.name,
            root=self.cfg.dataset.root,
            train=False,
            download=True,
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
        seed = getattr(self.cfg, 'seed', None)
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
                seed=seed,
            )
            self.clients.append(client)

    def train_and_evaluate(self) -> List[float]:
        """Run local training, aggregation, and evaluation."""
        # Local training and send updates
        logger.info("Starting local training across clients")

        for c in tqdm(self.clients, desc="Training", leave=True):
            c.training()
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

        try: 
            if self.codecarbon_tracker is not None:
                self.codecarbon_tracker.stop()
        except Exception:
            pass

        # Evaluate on a single held-out test dataset (no partitioning)
        loader = DataLoader(self.test_ds, batch_size=self.cfg.client.batch_size)

        accuracies = []
        for c in tqdm(random.sample(self.clients, 5), desc="Evaluating", leave=True):
            acc = c.evaluate(loader) * 100
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

        mean_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)

        logger.info(f"Accuracy: {mean_accuracy:.2f} ± {std_accuracy:.3f}")
        if self.wandb_run is not None:
            try:
                import wandb  # type: ignore

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
                            "codecarbon_duration_seconds": emissions.duration,
                        }
                    )
                if emissions is not None:
                    logger.info(
                        f"Energy: {emissions.energy_consumed:.6f} kWh, Emissions: {emissions.emissions:.6f} kgCO2, Training Duration: {emissions.duration:.6f} seconds"
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
