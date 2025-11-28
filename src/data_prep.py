import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


def generate_dataset(n_samples: int = 2000, n_features: int = 20) -> pd.DataFrame:
    """Generate a synthetic binary classification dataset and return as a DataFrame."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=10,
        n_redundant=4,
        n_repeated=0,
        n_classes=2,
        n_clusters_per_class=2,
        flip_y=0.02,
        class_sep=1.0,
        random_state=RANDOM_STATE,
    )

    columns = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=columns)
    df["target"] = y
    return df


def main() -> None:
    df = generate_dataset()
    output_path = DATA_DIR / "clean_data.csv"
    df.to_csv(output_path, index=False)
    print(f"[data_prep] Saved synthetic dataset to {output_path.resolve()}")
    print(f"[data_prep] Shape: {df.shape}")


if __name__ == "__main__":
    main()
