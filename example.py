# Example: Federated learning with two clients and CKKS encryption, using synthetic data.

import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import FakeData
from torchvision.transforms import ToTensor
from federated_rolann.federated.client import Client
from federated_rolann.federated.coordinator import Coordinator
from federated_rolann.datasets import prepare_splits

# 1) Import encryption functions from the library
from federated_rolann.encrypted import (
    create_context,
    serialize_context,
    deserialize_context,
)

# 2) Create CKKS context and serialize for coordinator and clients
master_ctx = create_context()

ctx_secret = serialize_context(master_ctx, secret_key=True)
ctx_public = serialize_context(master_ctx, secret_key=False)
client_ctx = deserialize_context(ctx_secret)
coord_ctx  = deserialize_context(ctx_public)

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

# 6) Instantiate coordinator and clients (encrypted=True)
coord = Coordinator(
    num_classes=10,
    device="cpu",
    num_clients=2,
    encrypted=True,
    ctx=coord_ctx,
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
        encrypted=True,
        ctx=client_ctx,
        broker="localhost",
        port=1883,
    )
    clients.append(c)

# 7) Local training and sending update
for c in clients:
    c.training() # Train on its local partition
    c.aggregate_parcial() # Send M/US to the coordinator

# The coordinator will automatically aggregate and publish the global model.

# 8) Give time for clients to receive the global model via MQTT
import time
time.sleep(2)

# 9) Evaluation on the full dataset
print("---- Global evaluation ----")
loader_train = DataLoader(datasets[0], batch_size=32)
loader_test  = DataLoader(datasets[1], batch_size=32)

for i, c in enumerate(clients):
    acc_train = c.evaluate(loader_train)
    acc_test  = c.evaluate(loader_test)
    print(f"Client {i}: train acc = {acc_train:.2f}, test acc = {acc_test:.2f}")
