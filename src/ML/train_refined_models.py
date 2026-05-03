from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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

        rows.append(
            {
                "threshold": threshold,
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred, zero_division=0),
                "roc_auc_from_prob": roc_auc_score(y_true, y_prob),
                "positive_prediction_rate": float(y_pred.mean()),
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


def build_logistic_feature_table(
    pipeline: Pipeline,
    feature_names: list[str],
) -> pd.DataFrame:
    model = pipeline.named_steps["logistic_regression"]

    coef_df = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": model.coef_[0],
        }
    )
    coef_df["abs_coefficient"] = coef_df["coefficient"].abs()
    coef_df = coef_df.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)
    return coef_df


def build_rf_feature_table(model: RandomForestClassifier, feature_names: list[str]) -> pd.DataFrame:
    imp_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
        }
    )
    imp_df = imp_df.sort_values("importance", ascending=False).reset_index(drop=True)
    return imp_df


def select_preferred_model(cv_summary: pd.DataFrame) -> dict:
    """
    Preference rule:
    1. Higher mean F1
    2. Then higher mean ROC-AUC
    3. Then higher mean Recall
    """
    ranked = cv_summary.sort_values(
        by=["f1_mean", "roc_auc_mean", "recall_mean"],
        ascending=False,
    ).reset_index(drop=True)

    best = ranked.iloc[0].to_dict()
    return best


def main() -> None:
    print("Starting refined model training and validation...")

    ml_dir = PROCESSED_DIR / "ml"
    input_path = ml_dir / "model_ready_baseline_rural_encoded.csv"
    results_dir = ml_dir / "refined_model_results"
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

    # Refined Logistic Regression: scaled pipeline
    print("\nRunning 5-fold CV for scaled Logistic Regression...")
    scaled_log_reg = Pipeline(
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

    log_cv = cross_validate(
        scaled_log_reg,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False,
    )

    log_summary = summarise_cv_results(log_cv, "scaled_logistic_regression")

    print("Generating out-of-fold probabilities for scaled Logistic Regression threshold tuning...")
    log_oof_prob = cross_val_predict(
        scaled_log_reg,
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

    # Fit on full data for feature interpretation
    print("\nFitting final full-data models for feature interpretation...")
    scaled_log_reg.fit(X, y)
    rf.fit(X, y)

    log_features = build_logistic_feature_table(scaled_log_reg, X.columns.tolist())
    rf_features = build_rf_feature_table(rf, X.columns.tolist())

    best_f1_row = threshold_results.sort_values("f1", ascending=False).iloc[0]
    best_recall_row = threshold_results.sort_values("recall", ascending=False).iloc[0]
    preferred_model = select_preferred_model(cv_summary)

    cv_summary_path = results_dir / "refined_model_cv_summary.csv"
    threshold_results_path = results_dir / "scaled_logistic_threshold_sweep.csv"
    log_oof_prob_path = results_dir / "scaled_logistic_oof_probabilities.csv"
    log_features_path = results_dir / "scaled_logistic_feature_effects.csv"
    rf_features_path = results_dir / "random_forest_feature_importance.csv"
    summary_json_path = results_dir / "refined_model_summary.json"

    cv_summary.to_csv(cv_summary_path, index=False)
    threshold_results.to_csv(threshold_results_path, index=False)
    log_features.to_csv(log_features_path, index=False)
    rf_features.to_csv(rf_features_path, index=False)

    pd.DataFrame(
        {
            "actual_target": y,
            "scaled_logistic_oof_probability": log_oof_prob,
        }
    ).to_csv(log_oof_prob_path, index=False)

    summary_payload = {
        "n_rows": int(len(df)),
        "n_features": int(len(X.columns)),
        "target_distribution": {str(k): int(v) for k, v in y.value_counts().to_dict().items()},
        "cv_summary": cv_summary.to_dict(orient="records"),
        "preferred_model": preferred_model,
        "best_scaled_logistic_threshold_by_f1": {
            "threshold": float(best_f1_row["threshold"]),
            "accuracy": float(best_f1_row["accuracy"]),
            "precision": float(best_f1_row["precision"]),
            "recall": float(best_f1_row["recall"]),
            "f1": float(best_f1_row["f1"]),
        },
        "best_scaled_logistic_threshold_by_recall": {
            "threshold": float(best_recall_row["threshold"]),
            "accuracy": float(best_recall_row["accuracy"]),
            "precision": float(best_recall_row["precision"]),
            "recall": float(best_recall_row["recall"]),
            "f1": float(best_recall_row["f1"]),
        },
    }

    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    print("\n--- Refined Cross-Validation Summary ---")
    print(cv_summary)

    print("\n--- Scaled Logistic Threshold Sweep ---")
    print(threshold_results)

    print("\nPreferred model based on CV:")
    print(preferred_model)

    print("\nBest scaled Logistic threshold by F1:")
    print(best_f1_row)

    print("\nBest scaled Logistic threshold by Recall:")
    print(best_recall_row)

    print("\nTop 10 scaled Logistic Regression features:")
    print(log_features.head(10))

    print("\nTop 10 Random Forest features:")
    print(rf_features.head(10))

    print("\nRefined model training and validation completed successfully.")
    print(f"CV summary CSV: {cv_summary_path}")
    print(f"Threshold sweep CSV: {threshold_results_path}")
    print(f"OOF probabilities CSV: {log_oof_prob_path}")
    print(f"Scaled logistic feature effects CSV: {log_features_path}")
    print(f"Random forest importance CSV: {rf_features_path}")
    print(f"Summary JSON: {summary_json_path}")


if __name__ == "__main__":
    main()