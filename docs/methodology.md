# Methodology Framework

Every case study in this repository follows the same 8-step framework, designed to mirror industry-standard ML/AI workflows:

## 1. Problem Framing
- Clearly state the business or scientific question
- Define a success metric upfront
- Establish a baseline benchmark to beat

## 2. Data Understanding
- Distribution checks (histograms, summary statistics)
- Missing value analysis
- Feature relationships (correlation, mutual information)
- Class balance (for classification)

## 3. Preprocessing
- Cleaning (missing values, outliers)
- Encoding (categorical → numeric)
- Scaling (standardization, normalization)
- Train/validation/test split with stratification

## 4. Baseline Model
- Always start with the simplest reasonable model
- Linear regression for regression
- Logistic regression / Naive Bayes for classification
- K-Means with default K for clustering

## 5. Advanced Model
- Iterate with more sophisticated approaches
- Tree-based ensembles (Random Forest, XGBoost)
- Deep learning where appropriate
- Hyperparameter tuning (Grid/Random/Bayesian Search)

## 6. Evaluation
- Multiple metrics, not just accuracy
- Cross-validation for robust estimates
- Error analysis — where does the model fail?
- Confidence intervals on metrics

## 7. Interpretation
- Feature importance (permutation, SHAP)
- Partial dependence plots
- Grad-CAM for image models
- Attention maps for transformers

## 8. Reflection
- What worked and what didn't
- Limitations of the analysis
- Deployment considerations
- Next steps for production

## Why This Framework?

Industry ML projects fail more often from poor problem framing or unrealistic deployment expectations than from model performance. This framework forces explicit thinking at every step, not just the modelling phase.
