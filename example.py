from fedhenet.runner import ExperimentRunner
from omegaconf import OmegaConf
from loguru import logger
import copy

base_cfg = OmegaConf.load("conf/config_cifar100.yaml")
# base_cfg = OmegaConf.load("conf/config.yaml")

seed = 42


def main():
    experiment_settings = [
        {"split": "dirichlet", "alpha": 1.0, "num_clients": 100},
        {"split": "dirichlet", "alpha": 0.1, "num_clients": 100},
        {"split": "single_class", "alpha": 1.0, "num_clients": 100},
    ]

    for params in experiment_settings:
        cfg = copy.deepcopy(base_cfg)
        cfg.seed = seed
        cfg.dataset.split = params["split"]
        cfg.dataset.alpha = params["alpha"]
        cfg.dataset.num_clients = params["num_clients"]
        cfg.communication.encrypted = True
        cfg.algorithm.use_float16 = True
        cfg.algorithm.compress = True
        cfg.logging.enable_wandb = True
        cfg.logging.enable_codecarbon = True
        cfg.dataset.subsample_fraction = 1.0

        try:
            runner = ExperimentRunner(cfg)
            runner.run()
        except Exception:
            logger.exception("Error occurred during experiment run")
            raise


if __name__ == "__main__":
    main()
