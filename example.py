# Example: Federated learning with two clients and CKKS encryption, using synthetic data.
from federated_rolann.cli import simulate_from_config

simulate_from_config("configs/sim_cifar_dirichlet.toml")
