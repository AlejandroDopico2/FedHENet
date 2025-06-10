import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import FakeData
from torchvision.transforms import ToTensor

# 1) Importar funciones de cifrado de la librería
from federated_rolann.encrypted import (
    create_context,
    serialize_context,
    deserialize_context,
)

# 2) Crear contexto CKKS y serializar para coordinador y clientes
master_ctx = create_context()

ctx_secret = serialize_context(master_ctx, secret_key=True)
ctx_public = serialize_context(master_ctx, secret_key=False)
client_ctx = deserialize_context(ctx_secret)
coord_ctx  = deserialize_context(ctx_public)

# 3) Importar clases federadas
from federated_rolann.federated.client import Cliente
from federated_rolann.federated.coordinator import Coordinador

# 4) Preparar datasets sintéticos
ds_full_train = FakeData(size=128, image_size=(3,224,224), num_classes=10, transform=ToTensor())
ds_test = FakeData(size=64,  image_size=(3,224,224), num_classes=10, transform=ToTensor())

# 5) Particionar el train en dos partes iguales (un cliente cada una)
ds1, ds2 = random_split(ds_full_train, [64, 64])

# 6) Instanciar coordinador y clientes (encrypted=True)
coord = Coordinador(
    num_classes=10,
    device="cpu",
    num_clients=2,
    encrypted=True,
    ctx=coord_ctx,
    broker="localhost",
    port=1883,
)

clients = []
for i, ds in enumerate((ds1, ds2)):
    c = Cliente(
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

# 7) Entrenamiento local y envío de actualización
for c in clients:
    c.training() # Entrena sobre su partición local
    c.aggregate_parcial() # Envía M/US al coordinador

# El coordinador agregará y publicará automáticamente el modelo global.

# 8) Dar tiempo a que los clientes reciban el modelo global por MQTT
import time
time.sleep(2)

# 9) Evaluación sobre el conjunto completo
print("---- Evaluación global ----")
loader_train = DataLoader(ds_full_train, batch_size=32)
loader_test  = DataLoader(ds_test, batch_size=32)

for i, c in enumerate(clients):
    acc_train = c.evaluate(loader_train)
    acc_test  = c.evaluate(loader_test)
    print(f"Cliente {i}: train acc = {acc_train:.2f}, test acc = {acc_test:.2f}")
