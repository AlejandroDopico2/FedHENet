# Example: Simple federated learning with two clients, no encryption, using synthetic data.
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import FakeData
from torchvision.transforms import ToTensor

from federated_rolann.federated.client import Client
from federated_rolann.federated.coordinator import Coordinator

# 1) Prepare synthetic datasets
ds_full_train = FakeData(size=64, image_size=(3, 32, 32), num_classes=5, transform=ToTensor())
ds_test = FakeData(size=32, image_size=(3, 32, 32), num_classes=5, transform=ToTensor())

# 2) Split the training dataset into two parts
ds1, ds2 = random_split(ds_full_train, [32, 32])

# 3) Instantiate coordinator and clients (without encryption)
coord = Coordinator(
    num_classes=5,
    device="cpu",
    num_clients=2,
    encrypted=False,
    ctx=None,
    broker="localhost",
    port=1883,
)

clients = []
for i, ds in enumerate((ds1, ds2)):
    c = Client(
        num_classes=5,
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
loader_train = DataLoader(ds_full_train, batch_size=16)
loader_test = DataLoader(ds_test, batch_size=16)

for i, c in enumerate(clients):
    acc_train = c.evaluate(loader_train)
    acc_test = c.evaluate(loader_test)
    print(f"Client {i}: acc train = {acc_train:.2f}, acc test = {acc_test:.2f}")
