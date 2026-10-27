# ReDeLEx

[![CI](https://github.com/jakubpeleska/ReDeLEx/actions/workflows/ci.yml/badge.svg?branch=develop&event=push)](https://github.com/jakubpeleska/ReDeLEx/actions/workflows/ci.yml?query=branch%3Adevelop)
[![website](https://img.shields.io/badge/website-live-brightgreen)](https://relational.fel.cvut.cz)
[![PyPI version](https://img.shields.io/pypi/v/redelex?color=brightgreen)](https://pypi.org/project/redelex/)
[![Python versions](https://img.shields.io/pypi/pyversions/redelex)](https://pypi.org/project/redelex/)
[![arXiv](https://img.shields.io/badge/arXiv-2506.22199-b31b1b.svg)](https://arxiv.org/abs/2506.22199)
[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://opensource.org/licenses/MIT)

**ReDeLEx** (Relational Deep Learning Exploration) is a Python framework for the development and evaluation of **Relational Deep Learning (RDL)** models. It enables end-to-end experimentation with graph-based neural networks on **relational databases (RDBs)**, building on the CTU Relational Learning Repository and fully integrating with the [RelBench](https://github.com/snap-stanford/relbench) interface.

It provides tools to transform SQL databases into heterogeneous graph representations suitable for Graph Neural Networks (GNNs), supports both static and temporal tasks, and enables a structured comparison across classical and deep learning models.

## 🧠 Features

- ✅ Supports **direct SQL database connectivity** (local or remote RDBMS)
- 🔗 Transforms relational schemas into **heterogeneous graphs**
- 🧩 Automates **attribute type inference** and encoding (categorical, numerical, text, time)
- 📦 Provides access to **70+ relational datasets** from the [CTU Repository](https://relational.fel.cvut.cz)
- 🧪 Supports **benchmarking tasks** including binary/multiclass classification, temporal tasks, and pretraining
- 🧠 Compatible with diverse neural architectures (e.g., GraphSAGE, Transformer-based models)
- 📊 Evaluates classical ML models (e.g., LightGBM, Propositionalization) alongside RDL models

## 📦 Installation

Install ReDeLEx via pip:

```bash
pip install redelex
```

## 🚀 Quickstart

### Loading CTU datasets

By name, using the RelBench-style registry:

```python
from redelex.datasets import get_dataset
from redelex.tasks import get_task, get_task_names

dataset = get_dataset('ctu-financial')
db = dataset.get_db()

get_task_names('ctu-financial')
task = get_task('ctu-financial', 'financial-original')
train_table = task.get_table('train')
```

The database is downloaded on the first `get_db()` call and cached as parquet files
in `~/.cache/redelex/<dataset>/db`; task tables are cached next to it. Set
`REDELEX_CACHE_DIR` before importing ReDeLEx to change the location, and delete the
cache directory to rebuild it.

Using ReDeLEx directly:

```python
from redelex import datasets as ctu_datasets

dataset = ctu_datasets.Financial(cache_dir='./cache/ctu-financial')
db = dataset.get_db()
```

Without a `cache_dir`, the database is downloaded again for every new dataset object.

### Loading a custom SQL database

```python
from redelex.datasets import DBDataset

custom_dataset = DBDataset(
    dialect="mariadb",  # e.g. postgresql, sqlite, mysql
    driver="mysqlconnector",
    user="your_user",
    password="your_password",
    host="your_host",
    port=3306,
    database="your_database"
)

db = custom_dataset.get_db(upto_test_timestamp=False)
```

Note: For full examples of task and schema configuration, see [examples in `ctu_datasets.py`](https://github.com/jakubpeleska/ReDeLEx/blob/main/redelex/datasets/ctu_datasets.py).

## 📚 Tasks & Use Cases

ReDeLEx supports:

- **Node-level prediction** (static or temporal)
- **Link prediction**
- **Pretraining tasks** via value masking
- **Database modification** for complex task generation

Each task is backed by a training table and optionally a temporal schema.

## 🏗️ Architecture

RDL models in ReDeLEx are modular and consist of:

1. **Attribute encoders** for tabular data
2. **Tabular models** (optional, e.g. ResNet)
3. **Graph Neural Network** layers
4. **Task-specific heads** (e.g. MLP classifiers)

Supported model examples include:

- Linear SAGE
- Tabular ResNet + GraphSAGE
- DBFormer (Transformer-based)

## 📈 Experiments

ReDeLEx includes tools for:

- Selecting RDL-suitable datasets based on structure and size
- Comparing RDL with traditional ML and propositionalization
- Benchmarking across 70+ relational datasets from various domains

For experimental results and performance benchmarks, see the [ECML PKDD 2025 paper](https://arxiv.org/abs/2506.22199).

## ⚙️ Development

### Install `uv` (recommended for managing environments)

- macOS & Linux

```bash
wget -qO- https://astral.sh/uv/install.sh | sh
```

- Windows

```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

More info: [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)

### Install dependencies

Requires Python 3.12 on Linux x86_64, macOS arm64 (Apple silicon) or Windows x86_64;
Intel Macs are not supported, as PyTorch 2.9 has no wheels for them. Each command
installs everything, including the compiled PyTorch Geometric extensions (`pyg-lib`,
`torch-scatter`, `torch-sparse`).

CPU (torch 2.9.1):

```bash
uv sync
```

CUDA 12.8 (torch 2.9.1):

```bash
uv sync --no-group cpu --group cu128
```

CUDA 12.4 (old CUDA env, torch 2.4.1):

```bash
uv sync --no-group cpu --group cu124
```

The CUDA groups are Linux/Windows only, because there is no CUDA on macOS; on macOS
they install a CPU torch without the PyTorch Geometric extensions, so use the CPU
group there.

### Run the tests

```bash
uv run pytest
```

Tests that download every CTU database and build every task are excluded by
default; run them explicitly with:

```bash
uv run pytest -m needs_network
```

Restrict them to a few databases with `REDELEX_SMOKE_DATASETS`:

```bash
REDELEX_SMOKE_DATASETS=ctu-financial,ctu-seznam uv run pytest -m needs_network
```

Every pull request and every push to `develop` runs the offline tests, the
linter and a packaging check. The smoke tests are not automatic - trigger them
from the **CTU smoke tests** workflow in the Actions tab, which takes the same
comma-separated list of databases.

### Enable and run `pre-commit`

```bash
uv run pre-commit install
uv run pre-commit run
```

### Database schema visualizations

Visualizations run on `Graphviz`, which needs to be available on your system.

- Install `Graphviz` https://graphviz.org/download/

## 📜 Citation

If you use ReDeLEx in your work, please cite:

```
@misc{peleska2025redelex,
  title={REDELEX: A Framework for Relational Deep Learning Exploration},
  author={Jakub Peleška and Gustav Šír},
  year={2025},
  eprint={2506.22199},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2506.22199},
}
```

## 📎 Acknowledgements

This project has received funding from the European Union’s Horizon Europe program under the grant agreement TUPLES No. 101070149, and the Czech Science Foundation grant No. 24-11664S.
