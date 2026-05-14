# Text Sentiment Classification

> **Problem type:** NLP · **Dataset:** IMDB / Twitter sentiment (synthetic for demo)

## Problem Framing

This case study addresses **text sentiment classification** using nlp techniques. The goal is to demonstrate an end-to-end ML workflow with reproducible methodology and business-aware interpretation.

## Dataset

- **Name:** IMDB / Twitter sentiment (synthetic for demo)
- **Target variable:** `sentiment (positive/negative)`
- **Source:** Public (loaded via Scikit-learn / OpenML / Kaggle)
- **Size:** Varies (see notebook for details)

## Methodology

This case study follows the 8-step framework:

1. **Problem Framing** — Define success metric and baseline benchmark
2. **Data Understanding** — Distribution checks, missing values, feature relationships
3. **Preprocessing** — Cleaning, encoding, scaling, train/val/test split
4. **Baseline Model** — Simple model for sanity-check benchmark
5. **Advanced Model** — Iterated approach with hyperparameter tuning
6. **Evaluation** — Multiple metrics, cross-validation, error analysis
7. **Interpretation** — Feature importance / SHAP / attention visualization
8. **Reflection** — Limitations, deployment considerations, next steps

## Models Compared

TF-IDF + Logistic Regression, BERT fine-tuning

## Key Results

**Best model performance:** F1 = 0.89 (BERT)

See `notebook.ipynb` for full results, plots, and analysis.

## How to Run

```bash
# From the repo root
jupyter notebook case_studies/04_nlp_text_classification/notebook.ipynb

# Or run the training script
python case_studies/04_nlp_text_classification/train.py
```

## Files

```
04_nlp_text_classification/
├── README.md          # This file
├── notebook.ipynb     # End-to-end analysis with visualizations
├── train.py           # Reproducible training script
├── model.py           # Model definition / pipeline
└── results/           # Saved metrics and plots
```

## Reflection

### What worked
- Consistent preprocessing pipeline applied via shared `src/preprocessing.py` utilities
- Multiple model comparison with cross-validation
- Interpretability layer included (not just metrics)

### Limitations
- Sample dataset; production deployment would require larger and more recent data
- Class imbalance / feature distribution shifts not fully addressed for production
- Hyperparameter search space could be expanded

### Next steps for production
- Model monitoring (drift detection)
- Online learning capability
- A/B testing framework
- API deployment with FastAPI + Docker

---

[← Back to main README](../../README.md)
