import argparse
import os
import logging
from loguru import logger

from .runner import ExperimentRunner

try:
    from omegaconf import DictConfig
    import hydra
    from hydra.core.config_store import ConfigStore
    HYDRA_AVAILABLE = True
except ImportError:
    DictConfig = dict  # Fallback for when Hydra is not available
    HYDRA_AVAILABLE = False

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

def main():
    """
    Legacy argparse-based CLI for backward compatibility.
    For new usage, prefer the Hydra-based CLI.
    """
    parser = argparse.ArgumentParser(prog="fedhenet", description="FedHeNet CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    sim = sub.add_parser(
        "simulate", help="Run single-machine simulation from a TOML config"
    )
    sim.add_argument("--config", required=True, help="Path to TOML config file")

    args = parser.parse_args()
    if args.command == "simulate":
        ExperimentRunner(args.config).run()

__all__ = ["main", "ExperimentRunner"]
