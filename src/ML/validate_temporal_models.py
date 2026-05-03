import json
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from config.paths import PROCESSED_DIR


TARGET_COL = "bank_access_deterioration_flag"
AUX_TARGET_COLS = [
    "bank_access_major_deterioration_flag",
    "bank_access_severe_deterioration_flag",
]
RANDOM_STATE = 42
N_SPLITS = 5


def evaluate_thresholds(
    y_true: pd.Series,
    y_prob: np.ndarray,
    thresholds: list[float],
) -> pd.DataFrame:
    rows = []

    actual_positive_count = int(y_true.sum())
    actual_positive_rate = float(y_true.mean())
    roc_auc = roc_auc_score(y_true, y_prob)

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)

        predicted_positive_count = int(y_pred.sum())
        predicted_positive_rate = float(y_pred.mean())

        rows.append(
            {
                "threshold": threshold,
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred, zero_division=0),
                "roc_auc_from_prob": roc_auc,
                "predicted_positive_count": predicted_positive_count,
                "predicted_positive_rate": predicted_positive_rate,
                "actual_positive_count": actual_positive_count,
                "actual_positive_rate": actual_positive_rate,
                "positive_count_gap": predicted_positive_count - actual_positive_count,
                "positive_rate_gap": predicted_positive_rate - actual_positive_rate,
            }
        )

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
    print("Starting temporal model validation...")

    ml_dir = PROCESSED_DIR / "ml"
    input_path = ml_dir / "model_ready_temporal_rural_encoded.csv"
    results_dir = ml_dir / "temporal_model_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading rural temporal encoded table from: {input_path}")
    df = pd.read_csv(input_path)

    print(f"Rows loaded: {len(df)}")
    print(f"Columns loaded: {len(df.columns)}")

    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COL}")

    drop_target_cols = [col for col in [TARGET_COL] + AUX_TARGET_COLS if col in df.columns]

    X = df.drop(columns=drop_target_cols).copy()
    y = df[TARGET_COL].copy()

    print(f"\nFeature rows: {len(X)}")
    print(f"Feature columns: {len(X.columns)}")

    print("\nTarget distribution:")
    print(y.value_counts(dropna=False))

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "logistic_regression",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    print("\nRunning 5-fold CV for scaled Logistic Regression...")
    cv_results = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False,
    )
    cv_summary = summarise_cv_results(cv_results, "scaled_logistic_regression")

    print("Generating out-of-fold probabilities for threshold tuning...")
    oof_prob = cross_val_predict(
        model,
        X,
        y,
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]

    threshold_grid = [round(x, 2) for x in np.arange(0.30, 0.81, 0.05)]
    threshold_results = evaluate_thresholds(y, oof_prob, threshold_grid)

    best_f1_row = threshold_results.sort_values(
        by=["f1", "recall", "precision"],
        ascending=[False, False, False],
    ).iloc[0]

    best_recall_row = threshold_results.sort_values(
        by=["recall", "precision", "f1"],
        ascending=[False, False, False],
    ).iloc[0]

    best_balance_row = threshold_results.iloc[
        (threshold_results["positive_rate_gap"].abs()).argmin()
    ]

    cv_summary_path = results_dir / "temporal_scaled_logistic_cv_summary.csv"
    threshold_results_path = results_dir / "temporal_scaled_logistic_threshold_sweep.csv"
    oof_probs_path = results_dir / "temporal_scaled_logistic_oof_probabilities.csv"
    summary_json_path = results_dir / "temporal_scaled_logistic_validation_summary.json"

    cv_summary.to_csv(cv_summary_path, index=False)
    threshold_results.to_csv(threshold_results_path, index=False)

    pd.DataFrame(
        {
            "actual_target": y,
            "scaled_logistic_oof_probability": oof_prob,
        }
    ).to_csv(oof_probs_path, index=False)

    summary_payload = {
        "n_rows": int(len(df)),
        "n_features": int(len(X.columns)),
        "target_distribution": {str(k): int(v) for k, v in y.value_counts().to_dict().items()},
        "cv_summary": cv_summary.to_dict(orient="records"),
        "best_threshold_by_f1": {
            "threshold": float(best_f1_row["threshold"]),
            "accuracy": float(best_f1_row["accuracy"]),
            "precision": float(best_f1_row["precision"]),
            "recall": float(best_f1_row["recall"]),
            "f1": float(best_f1_row["f1"]),
            "predicted_positive_count": int(best_f1_row["predicted_positive_count"]),
            "predicted_positive_rate": float(best_f1_row["predicted_positive_rate"]),
        },
        "best_threshold_by_recall": {
            "threshold": float(best_recall_row["threshold"]),
            "accuracy": float(best_recall_row["accuracy"]),
            "precision": float(best_recall_row["precision"]),
            "recall": float(best_recall_row["recall"]),
            "f1": float(best_recall_row["f1"]),
            "predicted_positive_count": int(best_recall_row["predicted_positive_count"]),
            "predicted_positive_rate": float(best_recall_row["predicted_positive_rate"]),
        },
        "closest_threshold_to_actual_positive_rate": {
            "threshold": float(best_balance_row["threshold"]),
            "accuracy": float(best_balance_row["accuracy"]),
            "precision": float(best_balance_row["precision"]),
            "recall": float(best_balance_row["recall"]),
            "f1": float(best_balance_row["f1"]),
            "predicted_positive_count": int(best_balance_row["predicted_positive_count"]),
            "predicted_positive_rate": float(best_balance_row["predicted_positive_rate"]),
        },
    }

    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    print("\n--- Temporal Logistic CV Summary ---")
    print(cv_summary)

    print("\n--- Temporal Logistic Threshold Sweep ---")
    print(threshold_results)

    print("\nBest temporal threshold by F1:")
    print(best_f1_row)

    print("\nBest temporal threshold by Recall:")
    print(best_recall_row)

    print("\nClosest temporal threshold to actual positive rate:")
    print(best_balance_row)

    print("\nTemporal model validation completed successfully.")
    print(f"CV summary CSV: {cv_summary_path}")
    print(f"Threshold sweep CSV: {threshold_results_path}")
    print(f"OOF probabilities CSV: {oof_probs_path}")
    print(f"Validation summary JSON: {summary_json_path}")


if __name__ == "__main__":
    main()