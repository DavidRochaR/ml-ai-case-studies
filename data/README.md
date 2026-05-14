# Data Directory

## Data Sources

Each case study loads data from one of:

| Case Study | Data Source |
|-----------|-------------|
| 01 Housing | `sklearn.datasets.fetch_california_housing()` |
| 02 Diabetes | OpenML `diabetes` dataset (Pima Indians) |
| 03 CNN CIFAR-10 | `torchvision.datasets.CIFAR10()` (auto-download) |
| 04 NLP | Synthetic — generated in notebook |
| 05 MNIST | `torchvision.datasets.MNIST()` (auto-download) |
| 06 Clustering | Synthetic via `sklearn.datasets.make_blobs()` |

All datasets are public and downloaded at runtime — no proprietary data committed to the repo.

## Reproducibility

Random seeds are fixed at `42` across all case studies for reproducible results.
