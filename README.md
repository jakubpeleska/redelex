# ReDeLEx

[![website](https://img.shields.io/badge/website-live-brightgreen)](https://relational.fel.cvut.cz)
[![PyPI version](https://img.shields.io/pypi/v/redelex?color=brightgreen)](https://pypi.org/project/redelex/)
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

If you're using [RelBench](https://github.com/snap-stanford/relbench), the CTU datasets are automatically supported.

## 🚀 Quickstart

### Loading CTU datasets

Using RelBench interface:

```python
from relbench.datasets import get_dataset
import redelex

dataset = get_dataset('ctu-seznam')
db = dataset.get_db()
```

Using ReDeLEx directly:

```python
from redelex import datasets as ctu_datasets

dataset = ctu_datasets.Seznam()
db = dataset.get_db()
```

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

For experimental results and performance benchmarks, see the [ECML PKDD 2025 paper](https://arxiv.org/abs/XXXX.XXXXX) _(coming soon)_.

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

CPU:

```bash
uv sync
uv pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cpu.html
```

CUDA 12.4:

```bash
uv sync --no-group cpu --group cu124
```

### Enable and run `pre-commit`

```bash
uv run pre-commit install
uv run pre-commit run
```

## 📜 Citation

If you use ReDeLEx in your work, please cite:

<!-- ```bibtex
@inproceedings{peleska2025redelex,
  title = {ReDeLEx: A Framework for Relational Deep Learning Exploration},
  author = {Peleška, Jakub and Šír, Gustav},
  booktitle = {ECML PKDD},
  year = {2025}
}
``` -->

## 📎 Acknowledgements

This project has received funding from the European Union’s Horizon Europe program under the grant agreement TUPLES No. 101070149, and the Czech Science Foundation grant No. 24-11664S.
