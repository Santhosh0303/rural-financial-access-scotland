from pathlib import Path
import json
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from config.paths import PROCESSED_DIR


TARGET_COL = "underserved_baseline"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def evaluate_classifier(model, X_test, y_test, model_name: str) -> tuple[dict, pd.DataFrame]:
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = None

    metrics = {
        "model_name": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob) if y_prob is not None else None,
    }

    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=["actual_0", "actual_1"],
        columns=["pred_0", "pred_1"],
    )
    cm_df.insert(0, "model_name", model_name)

    return metrics, cm_df


def build_logistic_feature_table(model, feature_names: list[str]) -> pd.DataFrame:
    coef_df = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": model.coef_[0],
        }
    )
    coef_df["abs_coefficient"] = coef_df["coefficient"].abs()
    coef_df = coef_df.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)
    return coef_df


def build_rf_feature_table(model, feature_names: list[str]) -> pd.DataFrame:
    imp_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
        }
    )
    imp_df = imp_df.sort_values("importance", ascending=False).reset_index(drop=True)
    return imp_df


def main() -> None:
    print("Starting baseline model training...")

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

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("\nTrain/test split summary:")
    print(f"X_train rows: {len(X_train)}")
    print(f"X_test rows: {len(X_test)}")
    print("y_train distribution:")
    print(y_train.value_counts(dropna=False))
    print("y_test distribution:")
    print(y_test.value_counts(dropna=False))

    # Logistic Regression
    print("\nTraining Logistic Regression...")
    log_reg = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=RANDOM_STATE,
    )
    log_reg.fit(X_train, y_train)
    log_metrics, log_cm = evaluate_classifier(log_reg, X_test, y_test, "logistic_regression")
    log_features = build_logistic_feature_table(log_reg, X.columns.tolist())

    # Random Forest
    print("Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_metrics, rf_cm = evaluate_classifier(rf, X_test, y_test, "random_forest")
    rf_features = build_rf_feature_table(rf, X.columns.tolist())

    metrics_df = pd.DataFrame([log_metrics, rf_metrics])
    confusion_df = pd.concat([log_cm, rf_cm], ignore_index=True)

    metrics_csv = results_dir / "baseline_model_metrics.csv"
    confusion_csv = results_dir / "baseline_model_confusion_matrices.csv"
    logistic_features_csv = results_dir / "logistic_regression_feature_effects.csv"
    rf_features_csv = results_dir / "random_forest_feature_importance.csv"
    summary_json = results_dir / "baseline_model_summary.json"

    metrics_df.to_csv(metrics_csv, index=False)
    confusion_df.to_csv(confusion_csv, index=False)
    log_features.to_csv(logistic_features_csv, index=False)
    rf_features.to_csv(rf_features_csv, index=False)

    summary_payload = {
        "input_rows": int(len(df)),
        "feature_count": int(len(X.columns)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "target_distribution": {str(k): int(v) for k, v in y.value_counts().to_dict().items()},
        "metrics": metrics_df.to_dict(orient="records"),
    }

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    print("\n--- Baseline Model Metrics ---")
    print(metrics_df)

    print("\n--- Confusion Matrices ---")
    print(confusion_df)

    print("\nTop 10 Logistic Regression features:")
    print(log_features.head(10))

    print("\nTop 10 Random Forest features:")
    print(rf_features.head(10))

    print("\nBaseline model training completed successfully.")
    print(f"Metrics CSV: {metrics_csv}")
    print(f"Confusion matrix CSV: {confusion_csv}")
    print(f"Logistic feature effects CSV: {logistic_features_csv}")
    print(f"Random forest importance CSV: {rf_features_csv}")
    print(f"Summary JSON: {summary_json}")


if __name__ == "__main__":
    main()