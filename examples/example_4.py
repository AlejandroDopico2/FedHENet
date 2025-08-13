# Example: Federated learning with three clients and CKKS encryption, using synthetic data.
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import FakeData
from torchvision.transforms import ToTensor

from federated_rolann.encrypted import (
    create_context,
    serialize_context,
    deserialize_context,
)
from federated_rolann.federated.client import Client
from federated_rolann.federated.coordinator import Coordinator

# Create CKKS context
master_ctx = create_context(poly_modulus_degree=8192, coeff_mod_bit_sizes=[40, 20, 40])
ctx_secret = serialize_context(master_ctx, secret_key=True)
ctx_public = serialize_context(master_ctx, secret_key=False)
client_ctx = deserialize_context(ctx_secret)
coord_ctx = deserialize_context(ctx_public)

# Synthetic dataset
ds_full_train = FakeData(size=90, image_size=(3, 16, 16), num_classes=4, transform=ToTensor())
ds_test = FakeData(size=30, image_size=(3, 16, 16), num_classes=4, transform=ToTensor())
ds1, ds2, ds3 = random_split(ds_full_train, [30, 30, 30])

# Instantiate coordinator and clients
coord = Coordinator(
    num_classes=4,
    device="cpu",
    num_clients=3,
    encrypted=True,
    ctx=coord_ctx,
    broker="localhost",
    port=1883,
)

clients = []
for i, ds in enumerate((ds1, ds2, ds3)):
    c = Client(
        num_classes=4,
        dataset=ds,
        device="cpu",
        client_id=i,
        encrypted=True,
        ctx=client_ctx,
        broker="localhost",
        port=1883,
    )
    clients.append(c)

for c in clients:
    c.training()
    c.aggregate_parcial()

import time
time.sleep(2)

print("---- Global evaluation ----")
loader_train = DataLoader(ds_full_train, batch_size=15)
loader_test = DataLoader(ds_test, batch_size=15)

for i, c in enumerate(clients):
    acc_train = c.evaluate(loader_train)
    acc_test = c.evaluate(loader_test)
    print(f"Client {i}: acc train = {acc_train:.2f}, acc test = {acc_test:.2f}")
