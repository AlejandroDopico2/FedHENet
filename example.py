from fedhenet.runner import ExperimentRunner
from omegaconf import OmegaConf
from loguru import logger

if __name__ == "__main__":
    # Load base config
    base_cfg = OmegaConf.load("conf/config.yaml")

    try:
        runner = ExperimentRunner(base_cfg)
        runner.run()
    except Exception:
        logger.exception("Error occurred during experiment run")
        raise
