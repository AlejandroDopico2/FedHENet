# rolann-federated

**Version:** 1.0.0  
**License:** MIT

Federated learning library that combines:
- **Pretrained ResNet18** (frozen) as feature extractor.  
- **ROLANN** as classifier.  
- Optional homomorphic encryption via CKKS (TenSEAL).  
- Communication between clients and coordinator via MQTT.

---

## Installation

```bash
git clone https://github.com/Ivanprdg/rolann-federated-lib.git
cd rolann-federated
pip install -e .
```
---

## Configuration

All main parameters can be adjusted:

| Parameter         | Description                                         | Default value      |
|-------------------|-----------------------------------------------------|--------------------|
| `broker`          | MQTT broker address                                 | `"localhost"`      |
| `port`            | MQTT broker port                                    | `1883`             |
| `num_clients`     | Number of clients to instantiate                    | `_`                |
| `num_classes`     | Number of classes in the dataset                    | `-`                |
| `device`          | PyTorch device (`"cpu"` or `"cuda"`)                | `"cpu"`            |
| `encrypted`       | Enable homomorphic encryption (True/False)          | `False`            |

Additionally, CKKS (TenSEAL) specific parameters are created in the script itself:

- `poly_modulus_degree`: Polynomial size (e.g. 8192).
- `coeff_mod_bit_sizes`: Coefficient sizes (e.g. `[60]`).
- `global_scale`: Global scale (e.g. `2**40`).

You can modify these values directly in the context creation block.

---

## Advanced: Injecting a custom `ROLANN` instance

By default, the coordinator and clients build an internal `ROLANN` with sensible defaults.  
If you need full control (e.g., custom hyper-parameters, encryption context), you can construct a `ROLANN` yourself and inject it into `Coordinator` and `Client`.

**Prerequisites (when using CKKS encryption):**
- The **coordinator** must have a **public** CKKS context (no secret key).
- Each **client** must have a **private** CKKS context (with secret key).
- `num_classes` of the injected `ROLANN` must match the constructor’s `num_classes`.

```python
from rolann_federated import ROLANN, Client, Coordinator
import tenseal as ts

# 1) (Optional) Create a CKKS context if you will encrypt M
ctx = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=8192,
    coeff_mod_bit_sizes=[60, 40, 40, 60],
)
ctx.generate_galois_keys()
ctx.generate_relin_keys()
ctx.global_scale = 2**40

# 2) Derive contexts for coordinator (public) and clients (private)
ctx_bytes = ctx.serialize(save_secret_key=True)
ctx_coord = ts.context_from(ctx_bytes)
ctx_coord.make_context_public()  # coordinator must NOT hold the secret key
ctx_client = ctx                  # client keeps the secret key

# 3) Build your own ROLANN objects (you can pass any custom options your ROLANN supports)
rolann_coord = ROLANN(num_classes=10, encrypted=True, context=ctx_coord)
rolann_client = ROLANN(num_classes=10, encrypted=True, context=ctx_client)

# 4) Inject them into Coordinator / Client
coord = Coordinator(
    num_classes=10,
    device="cuda",
    num_clients=4,
    broker="localhost",
    port=1883,
    rolann=rolann_coord,   # ← injected model
)

cli = Client(
    num_classes=10,
    dataset=train_loader,  # your Dataset/DataLoader
    device="cuda",
    client_id=1,
    broker="localhost",
    port=1883,
    rolann=rolann_client,  # ← injected model
)
```
---

## Usage example

1. **Start an MQTT broker** (for example Mosquitto):

2. **Run the included examples**:

   ```bash
   python example.py
   python example_2.py
   python example_3.py
   python example_4.py
   ```

   - `example.py`: Federated learning with two clients and CKKS encryption, using synthetic data.
   - `example_2.py`: Simple federated learning with two clients, no encryption, using synthetic data.
   - `example_3.py`: Local (non-federated) training of ROLANN using synthetic data and ResNet feature extraction.
   - `example_4.py`: Federated learning with three clients and CKKS encryption, using synthetic data.

   Each script demonstrates a different usage scenario for the library, from local training to federated learning with and without encryption.

---

You may also want to:
- **Modify the dataset or number of clients** in the examples to fit your use case.
- **Adjust CKKS parameters** for different security/performance tradeoffs.
- **Integrate your own data** by replacing the `FakeData` dataset with your own `torch.utils.data.Dataset`.

---
