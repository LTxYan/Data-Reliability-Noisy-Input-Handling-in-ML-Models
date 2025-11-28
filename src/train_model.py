from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "clean_data.csv"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


def load_data():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["target"])
    y = df["target"]
    return X, y


def train_models():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    log_reg = LogisticRegression(max_iter=200, random_state=RANDOM_STATE)
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    log_reg.fit(X_train_scaled, y_train)
    rf.fit(X_train, y_train)  # tree-based model works fine without scaling

    y_pred_lr = log_reg.predict(X_test_scaled)
    y_pred_rf = rf.predict(X_test)

    acc_lr = accuracy_score(y_test, y_pred_lr)
    acc_rf = accuracy_score(y_test, y_pred_rf)

    print(f"[train_model] Logistic Regression accuracy (clean): {acc_lr:.4f}")
    print(f"[train_model] Random Forest accuracy (clean):      {acc_rf:.4f}")

    # Save artifacts
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
    joblib.dump(log_reg, MODELS_DIR / "log_reg.joblib")
    joblib.dump(rf, MODELS_DIR / "random_forest.joblib")
    print(f"[train_model] Saved models to {MODELS_DIR.resolve()}")


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Run `python src/data_prep.py` first."
        )
    train_models()


if __name__ == "__main__":
    main()
