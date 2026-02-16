# Contributing to FedHENet

Thank you for your interest in contributing to **FedHENet**! We welcome contributions of all kinds — from bug fixes and documentation improvements to new algorithms and features.

---

## 📋 Table of Contents

- [How Can I Contribute?](#how-can-i-contribute)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Code Style](#code-style)
- [Submitting Changes](#submitting-changes)
- [Reporting Issues](#reporting-issues)
- [Areas We'd Love Help With](#areas-wed-love-help-with)

---

## How Can I Contribute?

There are many ways to contribute, regardless of your experience level:

- **Report bugs** — Found something broken? Open an issue.
- **Suggest features** — Have an idea? We'd love to hear it.
- **Fix bugs** — Browse open issues and submit a pull request.
- **Add new FL algorithms** — Implement additional federated learning methods.
- **Improve tests** — Increase coverage and add edge-case testing.
- **Write documentation** — Help make FedHENet more accessible.
- **Improve the library API** — Propose cleaner interfaces or refactors.

---

## Getting Started

### 1. Fork and clone the repository

```bash
git clone https://github.com/<your-username>/fedhenet.git
cd fedhenet
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install in development mode

```bash
pip install -e ".[dev]"
```

### 4. Verify your setup

```bash
python -m pytest tests/
```

---

## Development Workflow

1. **Create a branch** from `main` for your work:

   ```bash
   git checkout -b feature/my-new-feature
   ```

2. **Make your changes** — write code, tests, and documentation as needed.

3. **Run the linter** to ensure code quality:

   ```bash
   ruff check .
   ruff format .
   ```

4. **Run the tests** to make sure nothing is broken:

   ```bash
   python -m pytest tests/
   ```

5. **Commit** with a clear, descriptive message:

   ```bash
   git commit -m "feat: add FedXYZ algorithm with Dirichlet support"
   ```

6. **Push** your branch and open a Pull Request.

---

## Code Style

We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting. Please make sure your code passes before submitting:

```bash
ruff check .
ruff format .
```

General guidelines:

- Follow [PEP 8](https://peps.python.org/pep-0008/) conventions.
- Use type hints for function signatures.
- Write docstrings for all public classes and methods.
- Keep functions focused — prefer small, composable functions over monolithic ones.

---

## Submitting Changes

### Pull Request Process

1. Ensure your branch is up to date with `main`:

   ```bash
   git fetch origin
   git rebase origin/main
   ```

2. Open a Pull Request against `main` with:
   - A clear **title** summarizing the change.
   - A **description** explaining *what* and *why*.
   - References to any related **issues** (e.g., `Closes #42`).

3. A maintainer will review your PR. Please be patient — we may request changes or ask questions.

### Commit Message Convention

We loosely follow [Conventional Commits](https://www.conventionalcommits.org/):

| Prefix | Usage |
|--------|-------|
| `feat:` | New feature |
| `fix:` | Bug fix |
| `docs:` | Documentation only |
| `test:` | Adding or updating tests |
| `refactor:` | Code restructuring (no behavior change) |
| `chore:` | Build, CI, or tooling changes |

---

## Reporting Issues

When opening an issue, please include:

- **A clear title** describing the problem or request.
- **Steps to reproduce** (for bugs) with the config and command used.
- **Expected vs. actual behavior**.
- **Environment details**: OS, Python version, PyTorch version, GPU (if relevant).
- **Logs or error tracebacks** (if applicable).

---

## Areas We'd Love Help With

Here are some areas where contributions are especially welcome:

| Area | Description |
|------|-------------|
| **New FL Algorithms** | Implement methods like FedNova, SCAFFOLD, FedBN, etc., following the `BaseAlgorithm` interface in `fedhenet/algorithms/`. |
| **Testing** | Expand unit and integration tests in `tests/`. Add edge-case coverage for encryption, serialization, and non-IID splits. |
| **Library API** | Improve the public API ergonomics — cleaner constructors, better defaults, more Pythonic interfaces. |
| **Documentation** | Add usage guides, API reference (e.g., Sphinx/MkDocs), and tutorials. |
| **New Datasets** | Add support for more datasets beyond CIFAR-10/100. |
| **Benchmarks** | Add reproducible benchmark scripts and comparison tables. |
| **CI/CD** | Set up GitHub Actions for automated testing, linting, and publishing. |

### Adding a New Algorithm

To add a new federated learning algorithm:

1. Create a new file in `fedhenet/algorithms/` (e.g., `fednova.py`).
2. Subclass `BaseAlgorithm` and implement the required methods:

   ```python
   from .base import BaseAlgorithm

   class FedNova(BaseAlgorithm):
       def init_model(self, extractor=None):
           ...

       def local_train(self, model, loader, epochs=1, **kwargs):
           ...

       def compute_update(self, model):
           ...

       def serialize_update(self, update, encrypted, ctx=None):
           ...

       def aggregate_updates(self, updates):
           ...

       def apply_global(self, model, aggregated_update):
           ...
   ```

3. Register it in `fedhenet/algorithms/factory.py`:

   ```python
   from .fednova import FedNova

   ALGORITHM_REGISTRY["fednova"] = FedNova
   ```

4. Add tests and update the documentation.

---

## Code of Conduct

We are committed to providing a welcoming and inclusive environment. All contributors are expected to:

- Be respectful and constructive in all interactions.
- Welcome newcomers and help them get started.
- Focus on the work and avoid personal attacks.
- Report unacceptable behavior to the maintainers.

---

## Questions?

If you have questions about contributing, feel free to open a [Discussion](https://github.com/AlejandroDopico2/FedHENet/discussions) or reach out to the maintainers.

Thank you for helping make FedHENet better! 🚀
