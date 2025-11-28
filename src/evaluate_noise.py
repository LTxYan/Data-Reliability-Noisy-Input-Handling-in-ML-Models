from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "clean_data.csv"
MODELS_DIR = BASE_DIR / "models"

RANDOM_STATE = 42
NOISE_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4]  # std dev as fraction of feature std
MISSING_RATES = [0.0, 0.05, 0.1, 0.2]


def load_base_data():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["target"])
    y = df["target"]
    return train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )


def add_gaussian_noise(X: pd.DataFrame, level: float) -> pd.DataFrame:
    if level <= 0:
        return X.copy()
    X_noisy = X.copy()
    for col in X_noisy.columns:
        std = X_noisy[col].std()
        noise = np.random.normal(loc=0.0, scale=level * std, size=len(X_noisy))
        X_noisy[col] = X_noisy[col] + noise
    return X_noisy


def inject_missing_values(X: pd.DataFrame, rate: float) -> pd.DataFrame:
    if rate <= 0:
        return X.copy()
    X_mv = X.copy()
    n_rows, n_cols = X_mv.shape
    n_missing = int(rate * n_rows * n_cols)
    idx = np.random.randint(0, n_rows, size=n_missing)
    cols = np.random.randint(0, n_cols, size=n_missing)
    for r, c in zip(idx, cols):
        X_mv.iat[r, c] = np.nan
    # simple imputation: column mean
    X_mv = X_mv.fillna(X_mv.mean(numeric_only=True))
    return X_mv


def evaluate():
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    log_reg = joblib.load(MODELS_DIR / "log_reg.joblib")
    rf = joblib.load(MODELS_DIR / "random_forest.joblib")

    X_train, X_test, y_train, y_test = load_base_data()

    results_noise = []
    print("\n[Evaluate Noise] Gaussian feature noise")
    print("NoiseLevel | Acc_LogReg | Acc_RandomForest")
    for level in NOISE_LEVELS:
        X_test_noisy = add_gaussian_noise(X_test, level)
        X_test_noisy_scaled = scaler.transform(X_test_noisy)

        y_pred_lr = log_reg.predict(X_test_noisy_scaled)
        y_pred_rf = rf.predict(X_test_noisy)

        acc_lr = accuracy_score(y_test, y_pred_lr)
        acc_rf = accuracy_score(y_test, y_pred_rf)
        results_noise.append((level, acc_lr, acc_rf))
        print(f"{level:9.2f} | {acc_lr:10.4f} | {acc_rf:15.4f}")

    results_mv = []
    print("\n[Evaluate Missing Values] Random missing value injection + mean imputation")
    print("MissingRate | Acc_LogReg | Acc_RandomForest")
    for rate in MISSING_RATES:
        X_test_mv = inject_missing_values(X_test, rate)
        X_test_mv_scaled = scaler.transform(X_test_mv)

        y_pred_lr = log_reg.predict(X_test_mv_scaled)
        y_pred_rf = rf.predict(X_test_mv)

        acc_lr = accuracy_score(y_test, y_pred_lr)
        acc_rf = accuracy_score(y_test, y_pred_rf)
        results_mv.append((rate, acc_lr, acc_rf))
        print(f"{rate:10.2f} | {acc_lr:10.4f} | {acc_rf:15.4f}")

    # Plotting
    noise_levels, lr_noise, rf_noise = zip(*results_noise)
    mv_rates, lr_mv, rf_mv = zip(*results_mv)

    plt.figure(figsize=(8, 5))
    plt.plot(noise_levels, lr_noise, marker="o", label="LogReg - Gaussian noise")
    plt.plot(noise_levels, rf_noise, marker="s", label="RandomForest - Gaussian noise")
    plt.xlabel("Noise level (std fraction)")
    plt.ylabel("Accuracy")
    plt.title("Impact of Gaussian Noise on Model Accuracy")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(BASE_DIR / "noise_impact.png", dpi=200)
    print(f"\n[Evaluate] Saved plot to {BASE_DIR / 'noise_impact.png'}")


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Run `python src/data_prep.py` first."
        )
    evaluate()


if __name__ == "__main__":
    main()
