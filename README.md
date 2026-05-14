# 🧠 ML/AI Case Studies

> A curated portfolio of machine learning and AI experiments — regression, classification, deep learning, CNN, NLP, and unsupervised learning — documented end-to-end with reproducible notebooks and clean Python modules.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Made with Jupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange.svg)](https://jupyter.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-F7931E.svg)](https://scikit-learn.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)

---

## Overview

This repository consolidates hands-on ML/AI experiments developed during a Postgraduate Certificate in Big Data Analytics (Lambton College, 2024–2025) and personal practice. Each case study follows a consistent template: problem framing, data exploration, baseline model, advanced model, evaluation, and business interpretation.

**Why this matters:** Real-world ML deployment requires more than fitting a model — it requires understanding the data, communicating tradeoffs, and translating results into decisions. Each case study is structured to demonstrate that workflow.

**Built from real coursework + practice:** This portfolio consolidates and refines work originally distributed across multiple coursework repositories, restructured into a single, professional showcase aligned with industry standards.

## Case Studies

| # | Case Study | Problem Type | Tech Stack | Key Outcome |
|---|------------|--------------|------------|-------------|
| 01 | [Housing Price Prediction](case_studies/01_regression_housing/) | Regression | Scikit-learn, XGBoost | R² = 0.87, RMSE benchmarked vs Linear baseline |
| 02 | [Diabetes Risk Classification](case_studies/02_classification_diabetes/) | Binary Classification | Scikit-learn, SHAP | ROC-AUC = 0.83, top features identified |
| 03 | [Image Classification (CIFAR-10)](case_studies/03_cnn_image_classification/) | CNN / Computer Vision | PyTorch | 88% accuracy with custom CNN, transfer learning baseline |
| 04 | [Text Sentiment Classification](case_studies/04_nlp_text_classification/) | NLP | Scikit-learn, spaCy, Transformers | F1 = 0.89 with BERT fine-tuning |
| 05 | [Deep Learning Fundamentals](case_studies/05_deep_learning_neural_net/) | Neural Networks | PyTorch | Backprop from scratch, MNIST 98% |
| 06 | [Customer Segmentation](case_studies/06_clustering_unsupervised/) | Unsupervised Learning | Scikit-learn | K-Means + PCA, 4 actionable segments |

## Each Case Study Includes

```
case_studies/<name>/
├── README.md                  # Problem, data, methodology, results
├── notebook.ipynb             # End-to-end analysis with visualizations
├── train.py                   # Reproducible training script
├── model.py                   # Model definition / pipeline
└── results/                   # Saved metrics, plots, model artifacts
```

## How to Run

```bash
git clone https://github.com/DavidRochaR/ml-ai-case-studies.git
cd ml-ai-case-studies

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt

# Explore a specific case study
jupyter notebook case_studies/01_regression_housing/notebook.ipynb
```

## Methodology Framework

Every case study follows the same end-to-end framework:

1. **Problem Framing** — Business context, success metric, baseline benchmark
2. **Data Understanding** — Distribution checks, missing values, feature relationships
3. **Preprocessing** — Cleaning, encoding, scaling, train/val/test split
4. **Baseline Model** — Simple model for a sanity-check benchmark
5. **Advanced Model** — Iterated approach with hyperparameter tuning
6. **Evaluation** — Multiple metrics, cross-validation, residual/error analysis
7. **Interpretation** — Feature importance, SHAP, Grad-CAM, or attention maps
8. **Reflection** — Limitations, deployment considerations, next steps

## Tech Stack

`Python` `Jupyter` `Scikit-learn` `XGBoost` `LightGBM` `PyTorch` `TensorFlow`
`Pandas` `NumPy` `SciPy` `Matplotlib` `Seaborn` `Plotly`
`spaCy` `Transformers (Hugging Face)` `SHAP` `MLflow`

## Project Structure

```
ml-ai-case-studies/
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
├── case_studies/
│   ├── 01_regression_housing/
│   ├── 02_classification_diabetes/
│   ├── 03_cnn_image_classification/
│   ├── 04_nlp_text_classification/
│   ├── 05_deep_learning_neural_net/
│   └── 06_clustering_unsupervised/
├── src/
│   ├── __init__.py
│   ├── preprocessing.py        # Shared preprocessing utilities
│   ├── evaluation.py           # Shared evaluation utilities
│   └── visualization.py        # Shared plotting helpers
├── data/
│   └── README.md
├── docs/
│   └── methodology.md
└── tests/
    └── test_preprocessing.py
```

## Highlights

### What separates this portfolio
- **Consistent structure** — Every case study follows the same 8-step framework, making the work easy to navigate and compare.
- **Reproducible** — Scripts and notebooks both included; random seeds fixed; environment pinned via `requirements.txt`.
- **Business-aware** — Each case study ends with a "Reflection" section addressing real deployment considerations.
- **Interpretability-first** — SHAP, Grad-CAM, attention visualizations included where relevant. Not just "what" the model predicts, but "why."

### Cross-cutting themes
- Bias-variance tradeoff demonstrated across multiple problem types
- Class imbalance handling (SMOTE, class weights, focal loss)
- Model selection beyond accuracy (precision/recall tradeoffs, calibration)
- Feature engineering as a first-class citizen, not an afterthought

## Roadmap

- [x] Regression baseline (Housing)
- [x] Classification baseline (Diabetes)
- [x] CNN baseline (CIFAR-10)
- [x] NLP baseline (Sentiment)
- [x] Deep learning fundamentals
- [x] Unsupervised learning
- [ ] Time series forecasting (Prophet, LSTM)
- [ ] Recommender system (collaborative + content-based)
- [ ] Reinforcement learning (Q-learning, policy gradient)
- [ ] MLOps deployment (FastAPI + Docker + MLflow)

## Author

**David Rocha** — Geological Engineer & Renewable Energy PM | Big Data & AI
📍 Ontario, Canada
🔗 [LinkedIn](https://www.linkedin.com/in/davidrochar/) | [GitHub](https://github.com/DavidRochaR) | [Kaggle](https://www.kaggle.com/davidrochar9)

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
