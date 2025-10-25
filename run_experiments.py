from fedhenet.runner import ExperimentRunner
from omegaconf import OmegaConf
from loguru import logger
import copy

base_cfg = OmegaConf.load("conf/config.yaml")

seeds = [42]
algorithms = ["fedavg"]  # , "fedavg", "fedhenet"]
splits = [
    {"type": "dirichlet", "alpha": 100},
    {"type": "dirichlet", "alpha": 1.0},
    # {"type": "dirichlet", "alpha": 0.1},
    # {"type": "single_class", "alpha": 1.0},
]

num_clients_list = [10, 100]
encrypt_options = [True]
use_loggers = True

# rounds_options = [1, 5, 10]
rounds_options = [10]
epochs_options = [1]

already_done = set()

learning_rate = 0.01


def run_single(
    cfg, seed, algorithm, split_type, alpha, num_clients, encrypted, rounds, epochs
):
    cfg = copy.deepcopy(cfg)
    cfg.seed = seed
    cfg.algorithm.name = algorithm
    cfg.dataset.split = split_type
    cfg.dataset.alpha = alpha
    cfg.dataset.num_clients = num_clients
    cfg.communication.encrypted = encrypted
    cfg.logging.enable_wandb = use_loggers
    cfg.logging.enable_codecarbon = use_loggers
    cfg.dataset.subsample_fraction = 1

    cfg.algorithm.learning_rate = learning_rate

    key = (
        int(seed),
        str(algorithm),
        str(split_type),
        float(alpha),
        int(num_clients),
        bool(encrypted),
        int(rounds),
        int(epochs),
    )

    if key in already_done:
        logger.warning(
            f"Experiment already done: seed={seed}, algorithm={algorithm}, split={split_type}, alpha={alpha}, clients={num_clients}, encrypted={encrypted}, rounds={rounds}, epochs={epochs}"
        )
        return

    # Fair training setup
    if algorithm == "fedhenet":
        cfg.algorithm.num_rounds = 1
        cfg.algorithm.num_epochs = 1
    else:
        cfg.algorithm.num_rounds = rounds
        cfg.algorithm.num_epochs = epochs

    print(
        f"\n=== Running experiment | seed={seed} | algorithm={algorithm} | split={split_type} | "
        f"alpha={alpha} | clients={num_clients} | encrypted={encrypted} | rounds={cfg.algorithm.num_rounds} | "
        f"epochs={cfg.algorithm.num_epochs} ===\n"
    )

    try:
        runner = ExperimentRunner(cfg)
        runner.run()
        already_done.add(key)
    except Exception:
        logger.exception(
            f"Experiment failed: seed={seed}, algorithm={algorithm}, split={split_type}, "
            f"alpha={alpha}, clients={num_clients}, encrypted={encrypted}, "
            f"rounds={cfg.algorithm.num_rounds}, epochs={cfg.algorithm.num_epochs}"
        )
        pass


def main():
    for seed in seeds:
        for algorithm in algorithms:
            for s in splits:
                for num_clients in num_clients_list:
                    for encrypted in encrypt_options:
                        # FedHENet runs once; others test combinations
                        if algorithm == "fedhenet":
                            run_single(
                                cfg=base_cfg,
                                seed=seed,
                                algorithm=algorithm,
                                split_type=s["type"],
                                alpha=s["alpha"],
                                num_clients=num_clients,
                                encrypted=encrypted,
                                rounds=1,
                                epochs=1,
                            )
                        else:
                            for rounds in rounds_options:
                                for epochs in epochs_options:
                                    run_single(
                                        cfg=base_cfg,
                                        seed=seed,
                                        algorithm=algorithm,
                                        split_type=s["type"],
                                        alpha=s["alpha"],
                                        num_clients=num_clients,
                                        encrypted=encrypted,
                                        rounds=rounds,
                                        epochs=epochs,
                                    )


if __name__ == "__main__":
    main()
