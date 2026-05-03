from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from config.paths import PROCESSED_DIR


TARGET_COL = "underserved_baseline"
RANDOM_STATE = 42
N_SPLITS = 5


def evaluate_thresholds(y_true: pd.Series, y_prob: np.ndarray, thresholds: list[float]) -> pd.DataFrame:
    rows = []

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)

        row = {
            "threshold": threshold,
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "positive_prediction_rate": float(y_pred.mean()),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def summarise_cv_results(cv_results: dict, model_name: str) -> pd.DataFrame:
    metric_map = {
        "accuracy": "test_accuracy",
        "precision": "test_precision",
        "recall": "test_recall",
        "f1": "test_f1",
        "roc_auc": "test_roc_auc",
    }

    row = {"model_name": model_name}

    for metric_name, cv_key in metric_map.items():
        values = cv_results[cv_key]
        row[f"{metric_name}_mean"] = float(np.mean(values))
        row[f"{metric_name}_std"] = float(np.std(values))

    return pd.DataFrame([row])


def main() -> None:
    print("Starting baseline model validation...")

    ml_dir = PROCESSED_DIR / "ml"
    input_path = ml_dir / "model_ready_baseline_rural_encoded.csv"
    results_dir = ml_dir / "baseline_model_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading encoded model-ready table from: {input_path}")
    df = pd.read_csv(input_path)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COL}")

    print(f"Rows loaded: {len(df)}")
    print(f"Columns loaded: {len(df.columns)}")

    X = df.drop(columns=[TARGET_COL]).copy()
    y = df[TARGET_COL].copy()

    print(f"\nFeature rows: {len(X)}")
    print(f"Feature columns: {len(X.columns)}")
    print("\nTarget distribution:")
    print(y.value_counts(dropna=False))

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    # Logistic Regression
    print("\nRunning 5-fold CV for Logistic Regression...")
    log_reg = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=RANDOM_STATE,
    )

    log_cv = cross_validate(
        log_reg,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False,
    )

    log_summary = summarise_cv_results(log_cv, "logistic_regression")

    print("Generating out-of-fold probabilities for Logistic Regression threshold tuning...")
    log_oof_prob = cross_val_predict(
        log_reg,
        X,
        y,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]

    threshold_grid = [round(x, 2) for x in np.arange(0.30, 0.71, 0.05)]
    threshold_results = evaluate_thresholds(y, log_oof_prob, threshold_grid)

    # Random Forest
    print("Running 5-fold CV for Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    rf_cv = cross_validate(
        rf,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False,
    )

    rf_summary = summarise_cv_results(rf_cv, "random_forest")

    cv_summary = pd.concat([log_summary, rf_summary], ignore_index=True)

    best_f1_row = threshold_results.sort_values("f1", ascending=False).iloc[0]
    best_recall_row = threshold_results.sort_values("recall", ascending=False).iloc[0]

    cv_summary_path = results_dir / "baseline_model_cv_summary.csv"
    threshold_results_path = results_dir / "logistic_threshold_sweep.csv"
    log_oof_prob_path = results_dir / "logistic_oof_probabilities.csv"
    validation_summary_json = results_dir / "baseline_model_validation_summary.json"

    cv_summary.to_csv(cv_summary_path, index=False)
    threshold_results.to_csv(threshold_results_path, index=False)

    pd.DataFrame(
        {
            "actual_target": y,
            "logistic_oof_probability": log_oof_prob,
        }
    ).to_csv(log_oof_prob_path, index=False)

    summary_payload = {
        "n_rows": int(len(df)),
        "n_features": int(len(X.columns)),
        "target_distribution": {str(k): int(v) for k, v in y.value_counts().to_dict().items()},
        "cv_summary": cv_summary.to_dict(orient="records"),
        "best_f1_threshold": {
            "threshold": float(best_f1_row["threshold"]),
            "accuracy": float(best_f1_row["accuracy"]),
            "precision": float(best_f1_row["precision"]),
            "recall": float(best_f1_row["recall"]),
            "f1": float(best_f1_row["f1"]),
        },
        "best_recall_threshold": {
            "threshold": float(best_recall_row["threshold"]),
            "accuracy": float(best_recall_row["accuracy"]),
            "precision": float(best_recall_row["precision"]),
            "recall": float(best_recall_row["recall"]),
            "f1": float(best_recall_row["f1"]),
        },
    }

    with open(validation_summary_json, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    print("\n--- Cross-Validation Summary ---")
    print(cv_summary)

    print("\n--- Logistic Threshold Sweep ---")
    print(threshold_results)

    print("\nBest Logistic threshold by F1:")
    print(best_f1_row)

    print("\nBest Logistic threshold by Recall:")
    print(best_recall_row)

    print("\nBaseline model validation completed successfully.")
    print(f"CV summary CSV: {cv_summary_path}")
    print(f"Threshold sweep CSV: {threshold_results_path}")
    print(f"OOF probabilities CSV: {log_oof_prob_path}")
    print(f"Validation summary JSON: {validation_summary_json}")


if __name__ == "__main__":
    main()