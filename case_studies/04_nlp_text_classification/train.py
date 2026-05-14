"""NLP sentiment classification — training script."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.evaluation import evaluate_classification, compare_models
from model import get_models


def generate_synthetic_data(n=2000, seed=42):
    """Generate synthetic positive/negative review data."""
    np.random.seed(seed)
    pos_words = ["great", "amazing", "wonderful", "loved", "fantastic", "excellent",
                  "perfect", "best", "brilliant", "outstanding", "awesome", "superb"]
    neg_words = ["terrible", "awful", "worst", "disappointing", "bad", "horrible",
                  "boring", "waste", "poor", "weak", "frustrating", "annoying"]

    reviews, labels = [], []
    for _ in range(n // 2):
        n_words = np.random.randint(8, 25)
        text = " ".join(np.random.choice(pos_words, n_words // 2).tolist()
                        + ["movie", "film", "story", "acting"] * 2)
        reviews.append(text); labels.append(1)
    for _ in range(n // 2):
        n_words = np.random.randint(8, 25)
        text = " ".join(np.random.choice(neg_words, n_words // 2).tolist()
                        + ["movie", "film", "story", "acting"] * 2)
        reviews.append(text); labels.append(0)

    return pd.DataFrame({"text": reviews, "label": labels})


def main():
    data = generate_synthetic_data(n=2000)
    print(f"Dataset: {len(data)} samples, balanced classes")

    X_train, X_test, y_train, y_test = train_test_split(
        data["text"], data["label"], test_size=0.2, random_state=42, stratify=data["label"]
    )

    results = []
    for name, pipe in get_models().items():
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        results.append(evaluate_classification(y_test, preds, name=name))

    print("\n=== Model Comparison ===")
    print(compare_models(results).round(4))


if __name__ == "__main__":
    main()
