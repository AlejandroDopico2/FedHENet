# Example: Simple federated learning with two clients, no encryption, using synthetic data.
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import FakeData
from torchvision.transforms import ToTensor

from federated_rolann.federated.client import Client
from federated_rolann.federated.coordinator import Coordinator
from federated_rolann.datasets import prepare_splits

# 4) Prepare synthetic datasets
datasets = prepare_splits(
    name="cifar10",
    root="~/data",
    num_clients=3,
    split="iid",
    subsample_fraction=0.1,
    seed=42,
    train=True,
)


# 3) Instantiate coordinator and clients (without encryption)
coord = Coordinator(
    num_classes=10,
    device="cpu",
    num_clients=2,
    encrypted=False,
    ctx=None,
    broker="localhost",
    port=1883,
)

clients = []
for i, ds in enumerate(datasets):
    c = Client(
        num_classes=10,
        dataset=ds,
        device="cpu",
        client_id=i,
        encrypted=False,
        ctx=None,
        broker="localhost",
        port=1883,
    )
    clients.append(c)

# 4) Local training and sending update
for c in clients:
    c.training()
    c.aggregate_parcial()

import time
time.sleep(2)

# 5) Evaluation
print("---- Global evaluation ----")
loader_train = DataLoader(datasets[0], batch_size=32)
loader_test  = DataLoader(datasets[1], batch_size=32)

for i, c in enumerate(clients):
    acc_train = c.evaluate(loader_train)
    acc_test = c.evaluate(loader_test)
    print(f"Client {i}: acc train = {acc_train:.2f}, acc test = {acc_test:.2f}")
