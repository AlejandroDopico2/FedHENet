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
git clone https://github.com/Ivanprdg/Libreria_federated_rolann.git
cd rolann_federated
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
