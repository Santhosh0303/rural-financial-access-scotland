import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
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
    confusion_matrix,
)

from config.paths import PROCESSED_DIR


TARGET_COL = "bank_access_deterioration_flag"
AUX_TARGET_COLS = [
    "bank_access_major_deterioration_flag",
    "bank_access_severe_deterioration_flag",
]
RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS = 5


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


def evaluate_model(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    model_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics_df = pd.DataFrame(
        [
            {
                "model_name": model_name,
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, zero_division=0),
                "recall": recall_score(y_test, y_pred, zero_division=0),
                "f1": f1_score(y_test, y_pred, zero_division=0),
                "roc_auc": roc_auc_score(y_test, y_prob),
            }
        ]
    )

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    confusion_df = pd.DataFrame(
        [
            {
                "model_name": model_name,
                "actual_class": 0,
                "pred_0": int(tn),
                "pred_1": int(fp),
            },
            {
                "model_name": model_name,
                "actual_class": 1,
                "pred_0": int(fn),
                "pred_1": int(tp),
            },
        ]
    )

    return metrics_df, confusion_df, y_prob


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
    ranked = cv_summary.sort_values(
        by=["f1_mean", "roc_auc_mean", "recall_mean"],
        ascending=False,
    ).reset_index(drop=True)
    return ranked.iloc[0].to_dict()


def main() -> None:
    print("Starting temporal model training...")

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

    if "bank_access_severe_deterioration_flag" in df.columns:
        print("\nSevere target distribution:")
        print(df["bank_access_severe_deterioration_flag"].value_counts(dropna=False))

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

    random_forest = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    print("\nTraining scaled Logistic Regression...")
    log_metrics, log_confusion, log_test_prob = evaluate_model(
        scaled_log_reg,
        X_train,
        X_test,
        y_train,
        y_test,
        "scaled_logistic_regression",
    )

    print("Training Random Forest...")
    rf_metrics, rf_confusion, rf_test_prob = evaluate_model(
        random_forest,
        X_train,
        X_test,
        y_train,
        y_test,
        "random_forest",
    )

    test_metrics = pd.concat([log_metrics, rf_metrics], ignore_index=True)
    confusion_matrices = pd.concat([log_confusion, rf_confusion], ignore_index=True)

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    print("\nRunning 5-fold CV for scaled Logistic Regression...")
    log_cv = cross_validate(
        scaled_log_reg,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False,
    )
    log_cv_summary = summarise_cv_results(log_cv, "scaled_logistic_regression")

    print("Running 5-fold CV for Random Forest...")
    rf_cv = cross_validate(
        random_forest,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False,
    )
    rf_cv_summary = summarise_cv_results(rf_cv, "random_forest")

    cv_summary = pd.concat([log_cv_summary, rf_cv_summary], ignore_index=True)
    preferred_model = select_preferred_model(cv_summary)

    print("\nFitting full-data models for feature interpretation...")
    scaled_log_reg.fit(X, y)
    random_forest.fit(X, y)

    log_feature_effects = build_logistic_feature_table(scaled_log_reg, X.columns.tolist())
    rf_feature_importance = build_rf_feature_table(random_forest, X.columns.tolist())

    test_predictions = pd.DataFrame(
        {
            "actual_target": y_test.reset_index(drop=True),
            "scaled_logistic_probability": log_test_prob,
            "random_forest_probability": rf_test_prob,
        }
    )
    test_predictions["scaled_logistic_predicted_class"] = (
        test_predictions["scaled_logistic_probability"] >= 0.5
    ).astype(int)
    test_predictions["random_forest_predicted_class"] = (
        test_predictions["random_forest_probability"] >= 0.5
    ).astype(int)

    test_metrics_path = results_dir / "temporal_model_test_metrics.csv"
    confusion_path = results_dir / "temporal_model_confusion_matrices.csv"
    cv_summary_path = results_dir / "temporal_model_cv_summary.csv"
    log_features_path = results_dir / "temporal_scaled_logistic_feature_effects.csv"
    rf_features_path = results_dir / "temporal_random_forest_feature_importance.csv"
    test_predictions_path = results_dir / "temporal_model_test_predictions.csv"
    summary_json_path = results_dir / "temporal_model_summary.json"

    test_metrics.to_csv(test_metrics_path, index=False)
    confusion_matrices.to_csv(confusion_path, index=False)
    cv_summary.to_csv(cv_summary_path, index=False)
    log_feature_effects.to_csv(log_features_path, index=False)
    rf_feature_importance.to_csv(rf_features_path, index=False)
    test_predictions.to_csv(test_predictions_path, index=False)

    summary_payload = {
        "n_rows": int(len(df)),
        "n_features": int(len(X.columns)),
        "target_distribution": {str(k): int(v) for k, v in y.value_counts().to_dict().items()},
        "test_metrics": test_metrics.to_dict(orient="records"),
        "cv_summary": cv_summary.to_dict(orient="records"),
        "preferred_model": preferred_model,
    }

    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    print("\n--- Temporal Model Test Metrics ---")
    print(test_metrics)

    print("\n--- Temporal Model Confusion Matrices ---")
    print(confusion_matrices)

    print("\n--- Temporal Model CV Summary ---")
    print(cv_summary)

    print("\nPreferred model based on CV:")
    print(preferred_model)

    print("\nTop 10 scaled Logistic Regression features:")
    print(log_feature_effects.head(10))

    print("\nTop 10 Random Forest features:")
    print(rf_feature_importance.head(10))

    print("\nTemporal model training completed successfully.")
    print(f"Test metrics CSV: {test_metrics_path}")
    print(f"Confusion matrices CSV: {confusion_path}")
    print(f"CV summary CSV: {cv_summary_path}")
    print(f"Scaled logistic feature effects CSV: {log_features_path}")
    print(f"Random forest importance CSV: {rf_features_path}")
    print(f"Test predictions CSV: {test_predictions_path}")
    print(f"Summary JSON: {summary_json_path}")


if __name__ == "__main__":
    main()