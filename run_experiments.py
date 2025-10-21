from fedhenet.runner import ExperimentRunner
from omegaconf import OmegaConf
from loguru import logger
import copy
import time

base_cfg = OmegaConf.load("conf/config.yaml")

# seeds = [42, 123, 999]
seeds = [42]

splits = [
    # {"type": "dirichlet", "alpha": 1.0}, 
    {"type": "dirichlet", "alpha": 0.1}, 
    {"type": "dirichlet", "alpha": 100}, 
    {"type": "iid", "alpha": None},      
]

extra_client_runs = [10, 100]
encrypt_options = [False, True]

def run_single(cfg, seed, split_type, alpha, num_clients, encrypted):
    cfg = copy.deepcopy(cfg)
    cfg.seed = seed
    cfg.dataset.split = split_type
    cfg.dataset.alpha = alpha
    cfg.dataset.num_clients = num_clients
    cfg.communication.encrypted = encrypted
    cfg.logging.enable_wandb = True
    cfg.logging.enable_codecarbon = True
    cfg.dataset.subsample_fraction = 1

    print(
        f"\n=== Running experiment | seed={seed} | split={split_type} | "
        f"alpha={alpha} | clients={num_clients} | encrypted={encrypted} ===\n"
    )

    try:
        runner = ExperimentRunner(cfg)
        runner.run()
    except Exception as e:
        logger.exception(f"Experiment failed: seed={seed}, split={split_type}, alpha={alpha}, clients={num_clients}, encrypted={encrypted}")
        pass

def main():
    for seed in seeds:
        for s in splits:
            for clients in extra_client_runs:
                for encrypted in encrypt_options:
                    try:
                        run_single(
                            cfg=base_cfg,
                            seed=seed,
                            split_type=s["type"],
                            alpha=s["alpha"],
                            num_clients=clients,
                            encrypted=encrypted,
                        )
                    except Exception:
                        pass
            

if __name__ == "__main__":
    main()
