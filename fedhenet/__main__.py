#!/usr/bin/env python3
"""
Entry point for running FedHeNet as a module.

Usage:
    python -m fedhenet dataset.num_clients=50
    python -m fedhenet --config-name=config dataset.split=dirichlet dataset.alpha=0.3
"""

import hydra
from omegaconf import DictConfig
from .runner import ExperimentRunner


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Hydra-based entry point for experiments."""
    runner = ExperimentRunner(cfg)
    runner.run()


if __name__ == "__main__":
    main()
