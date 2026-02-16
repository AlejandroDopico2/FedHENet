<div align="center">

# 🧠 FedHENet

### **Frugal Federated Learning for Heterogeneous Environments**

*Official implementation of the paper accepted at **ESANN 2026***

**[Alejandro Dopico-Castro](https://github.com/AlejandroDopico2), [Oscar Fontenla-Romero](https://scholar.google.com/citations?user=J1gBZh0AAAAJ&hl=en), [Bertha Guijarro-Berdiñas](https://scholar.google.com/citations?user=4uEW-IoAAAAJ&hl=en), [Amparo Alonso-Betanzos](https://scholar.google.com/citations?user=4SX-5-oAAAAJ&hl=en), [Iván Pérez Digón](https://github.com/Ivanprdg)**

[![Python 3.10+](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D2.8-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv:2602.13024-b31b1b.svg)](https://arxiv.org/abs/2602.13024)
[![Code style: Ruff](https://img.shields.io/badge/Code%20style-Ruff-D7FF64?logo=ruff&logoColor=black)](https://docs.astral.sh/ruff/)

---

**FedHENet** is a *one-shot*, *privacy-preserving* federated learning framework that replaces
iterative gradient-based training with an **analytical closed-form solution**, drastically reducing
communication rounds, energy consumption, and carbon footprint — all while maintaining competitive accuracy.

[Key Features](#-key-features) •
[Installation](#-installation) •
[Quick Start](#-quick-start) •
[Configuration](#%EF%B8%8F-configuration-hydra) •
[Results](#-experimental-results) •
[Citation](#-citation)

</div>

---

## 📖 Introduction

Conventional federated learning (FL) methods such as **FedAvg** and **FedProx** require numerous communication rounds between a central coordinator and distributed clients, each round involving costly gradient computations, model serialization, and network transfers. In heterogeneous (non-IID) settings, these iterative methods often suffer from **client drift**, leading to degraded convergence and wasted resources.

**FedHENet** takes a fundamentally different approach. By leveraging an **analytical learning rule** (ROLANN) on top of a frozen pretrained feature extractor, each client computes a *closed-form update* in a **single round** — no iterative optimization, no hyperparameter tuning, no convergence issues. The coordinator aggregates these updates via **exact algebraic fusion**, guaranteeing a global model equivalent to one trained on the union of all client datasets.

> **Frugal by design:** One communication round. No learning rate. No epochs. No client drift. Up to **70% less energy** and **42% fewer bytes** transmitted compared to iterative baselines.

---

## ✨ Key Features

- **🔄 One-Shot Federated Learning** — Single communication round between clients and coordinator; no iterative training required.
- **🔒 Privacy-Preserving via Homomorphic Encryption** — Optional CKKS-based encryption (TenSEAL) enables secure aggregation without exposing raw model updates.
- **🌱 Frugal & Green AI** — Dramatically lower energy consumption and carbon footprint, tracked via [CodeCarbon](https://codecarbon.io/) integration.
- **🌐 Heterogeneous Environment Support** — Robust to non-IID data distributions (Dirichlet, single-class) and scalable to hundreds of clients.
- **⚙️ Hyperparameter-Free Analytical Learning** — No learning rate, momentum, or epoch tuning — the closed-form solution is computed directly.
- **📡 MQTT-Based Communication** — Lightweight publish/subscribe protocol for efficient coordinator–client messaging.
- **📊 Experiment Tracking** — Built-in [Weights & Biases](https://wandb.ai/) integration for logging and visualization.

---

## 🏗️ Pipeline overview

1. Each client extracts features locally using a **frozen pretrained ResNet-18** backbone.
2. A **ROLANN analytical head** computes closed-form weight matrices from the local data — no gradient descent.
3. Updates are (optionally) **encrypted with CKKS** homomorphic encryption and published to an MQTT broker.
4. The **coordinator** collects all client updates and performs **exact algebraic aggregation** in a single round.
5. The resulting global model is broadcast back to all clients.

---

## 📦 Installation

### Prerequisites

- Python ≥ 3.10
- An MQTT broker (e.g., [Mosquitto](https://mosquitto.org/)) running locally or remotely

### Install from source

```bash
git clone https://github.com/AlejandroDopico2/FedHENet.git
cd fedhenet
pip install .
```

### Development install

```bash
pip install -e ".[dev]"
```

> **Note:** [TenSEAL](https://github.com/OpenMined/TenSEAL) (for homomorphic encryption) requires a working C++ compiler. On Ubuntu: `sudo apt install build-essential cmake`.

---

## 🚀 Quick Start

### 1. Start the MQTT broker
FedHENet requires an active MQTT broker for communication. You can run Mosquitto locally:
```bash
# Using Mosquitto
mosquitto -v
```
> (Alternatively, you can use a Docker container: docker run -p 1883:1883 eclipse-mosquitto)


### 2. Run an experiment with Hydra

```bash
# Run with default config (CIFAR-10, FedHENet, 10 clients)
python -m fedhenet

# Override parameters from the command line
python -m fedhenet algorithm.name=fedhenet dataset.num_clients=100 dataset.alpha=0.1

# Use a different config file
python -m fedhenet --config-name=config_cifar100
```

### 3. Run a batch of experiments

```bash
python run_experiments.py
```

---

## ⚙️ Configuration (Hydra)

FedHENet uses [Hydra](https://hydra.cc/) for hierarchical configuration management. All settings are defined in YAML files under the `conf/` directory.

### Default configuration (`conf/config.yaml`)

```yaml
seed: 42

dataset:
  name: cifar10              # Dataset: cifar10, cifar100
  root: ./data
  split: dirichlet           # Partitioning: dirichlet, single_class
  alpha: 1.0                 # Dirichlet concentration (lower = more heterogeneous)
  num_clients: 10
  subsample_fraction: 1.0    # Fraction of data to use (useful for debugging)

coordinator:
  num_classes: 10
  device: cuda

client:
  device: cuda
  batch_size: 128

extractor:
  type: resnet18             # Pretrained feature extractor
  weights: default           # Use default pretrained weights

algorithm:
  name: fedhenet             # Algorithm: fedhenet, fedavg, fedprox
  compress: true             # Enable zlib compression
  use_float16: true          # Use FP16 for communication efficiency
  # For iterative baselines (fedavg, fedprox):
  # num_rounds: 10
  # num_epochs: 1
  # learning_rate: 0.01

communication:
  encrypted: true            # Enable CKKS homomorphic encryption
  broker: localhost           # MQTT broker address
  port: 1883                 # MQTT broker port

logging:
  enable_wandb: true         # Log to Weights & Biases
  wandb_project: fedhenet
  enable_codecarbon: true    # Track energy consumption
```

### Override examples

```bash
# Non-IID setting with 100 clients and encryption disabled
python -m fedhenet dataset.num_clients=100 dataset.alpha=0.1 communication.encrypted=false

# Run FedAvg baseline for comparison
python -m fedhenet algorithm.name=fedavg algorithm.num_rounds=10 algorithm.num_epochs=1

# CIFAR-100 experiment
python -m fedhenet --config-name=config_cifar100 dataset.num_clients=100
```

---

## 📊 Experimental Results

FedHENet is designed for Extreme Heterogeneity and Sustainability. Unlike iterative methods (FedAvg, FedProx), our analytical approach remains stable regardless of data distribution.

1. Robustness to Data Heterogeneity

We compare accuracy in the most challenging scenario (Single-class per client, N=10):

|Method	|CIFAR-10 Acc.|
|:--------|-------------:|
|FedAvg| 33.26%|
|FedProx| 40.75%|
|FedHENet (Ours) | 83.65%|

>  FedHENet is immune to client drift. While traditional FL collapses in heterogeneous settings, FedHENet maintains near-constant performance.

### 2. Green AI: Energy & Communication 🌿

Comparison for CIFAR-10 ($N=100$) using [CodeCarbon](https://codecarbon.io/):

| Metric | FedAvg | **FedHENet** | Reduction |
| :--- | :---: | :---: | :---: |
| **Energy Consumption** | 40.6 Wh | **11.2 Wh** | **-72.4%** |
| **Wall-clock Time** | 12.1 min | **3.7 min** | **-69.4%** |
| **Network Traffic** | 3.37 GB | **2.85 GB** | **-15.4%** |

> **Environmental Impact:** FedHENet drastically cuts the carbon footprint by eliminating iterative training and reducing the total uptime of client devices.
---

## 📂 Project Structure

```
fedhenet/
├── conf/                        # Hydra YAML configuration files
│   ├── config.yaml              # Default config (CIFAR-10)
│   └── config_cifar100.yaml     # CIFAR-100 config
├── fedhenet/                    # Main package
│   ├── algorithms/              # FL algorithms (FedAvg, FedProx, FedHENet)
│   ├── communication/           # MQTT transport layer
│   ├── crypto/                  # CKKS homomorphic encryption utilities
│   ├── federated/               # Coordinator & Client logic
│   ├── metrics/                 # Evaluation & energy tracking
│   ├── models/                  # Feature extractors
│   ├── runner/                  # Experiment orchestration
│   ├── rolann.py                # ROLANN analytical classifier
│   └── __main__.py              # Hydra CLI entry point
├── run_experiments.py           # Batch experiment launcher
├── pyproject.toml               # Package metadata & dependencies
└── LICENSE                      # MIT License
```

---

## 📝 Citation

If you use FedHENet in your research, please cite our paper:

```bibtex
@misc{dopico2026fedhenet,
      title={FedHENet: A Frugal Federated Learning Framework for Heterogeneous Environments}, 
      author={Alejandro Dopico-Castro and Oscar Fontenla-Romero and Bertha Guijarro-Berdiñas and Amparo Alonso-Betanzos and Iván Pérez Digón},
      year={2026},
      eprint={2602.13024},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2602.13024}, 
}
```

---

## 🤝 Contributing

We welcome contributions! Whether it's new FL algorithms, better tests, documentation, or API improvements — check out our [Contributing Guide](CONTRIBUTING.md) to get started.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙏 Acknowledgments

This work was developed at the [Universidade da Coruña (UDC)](https://www.udc.es/), [CITIC](https://citic.udc.es/). We thank the open-source communities behind [PyTorch](https://pytorch.org/), [TenSEAL](https://github.com/OpenMined/TenSEAL), [Hydra](https://hydra.cc/), [CodeCarbon](https://codecarbon.io/), and [Eclipse Mosquitto](https://mosquitto.org/).

<div align="center">

---

*Made with ❤️ for sustainable and private AI*

</div>
