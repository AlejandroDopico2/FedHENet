from fedhenet.runner import ExperimentRunner
from omegaconf import OmegaConf
from loguru import logger
import copy

base_cfg = OmegaConf.load("conf/config.yaml")

seed = 42

def main():
    cfg = copy.deepcopy(base_cfg)
    cfg.seed = seed
    cfg.dataset.split = "iid"
    cfg.dataset.num_clients = 2
    cfg.communication.encrypted = False
    cfg.logging.enable_wandb = True
    cfg.logging.enable_codecarbon = True
    cfg.dataset.subsample_fraction = 0.05

    try:
        runner = ExperimentRunner(cfg)
        runner.run()
    except Exception as e:
        logger.exception("Error occurred during experiment run")
        raise

if __name__ == "__main__":
    main()
