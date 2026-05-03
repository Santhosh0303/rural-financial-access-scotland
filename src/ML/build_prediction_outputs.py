from pathlib import Path
import json
import numpy as np
import pandas as pd
import geopandas as gpd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from config.paths import PROCESSED_DIR
from src.ML.build_model_ready_baseline import build_model_ready_table, TARGET_COL


RANDOM_STATE = 42


def load_refined_model_choices(results_dir: Path) -> tuple[str, float]:
    """
    Load the preferred model name and best scaled-logistic threshold from the refined
    validation summary. Falls back to sensible defaults if the file is missing.
    """
    summary_path = results_dir / "refined_model_summary.json"

    preferred_model_name = "random_forest"
    best_logistic_threshold = 0.60

    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        preferred_model_name = payload.get("preferred_model", {}).get("model_name", "random_forest")
        best_logistic_threshold = payload.get(
            "best_scaled_logistic_threshold_by_f1", {}
        ).get("threshold", 0.60)

    return preferred_model_name, float(best_logistic_threshold)


def assign_probability_band(prob_series: pd.Series) -> pd.Series:
    """
    Create a 3-tier probability band for dashboard-style display.
    Uses tertiles so the rural zones are split into low / medium / high risk groups.
    """
    ranked = prob_series.rank(method="first")
    bands = pd.qcut(
        ranked,
        q=3,
        labels=["low", "medium", "high"],
    )
    return bands.astype(str)


def main() -> None:
    print("Starting build_prediction_outputs...")

    ml_dir = PROCESSED_DIR / "ml"
    results_dir = ml_dir / "refined_model_results"
    output_dir = ml_dir / "prediction_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    master_features_path = ml_dir / "ml_features_baseline_2022.csv"
    geography_path = PROCESSED_DIR / "geography" / "zones_master_2022.gpkg"

    print(f"Loading ML master table from: {master_features_path}")
    master_df = pd.read_csv(master_features_path)

    print(f"Loading geography from: {geography_path}")
    zones_master_2022 = gpd.read_file(geography_path, layer="zones_master_2022")

    preferred_model_name, best_logistic_threshold = load_refined_model_choices(results_dir)

    print(f"Preferred model from refined validation: {preferred_model_name}")
    print(f"Best scaled logistic threshold by F1: {best_logistic_threshold}")

    print("\nRebuilding readable and encoded rural model tables from master data...")
    encoded_df, readable_df = build_model_ready_table(master_df)

    if len(encoded_df) != len(readable_df):
        raise ValueError("Readable and encoded rural tables do not have matching row counts.")

    X = encoded_df.drop(columns=[TARGET_COL]).copy()
    y = encoded_df[TARGET_COL].copy()

    print(f"Encoded rural rows: {len(encoded_df)}")
    print(f"Readable rural rows: {len(readable_df)}")
    print(f"Feature columns: {len(X.columns)}")

    # Fit scaled logistic regression on full rural data
    print("\nFitting scaled Logistic Regression on full rural data...")
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
    scaled_log_reg.fit(X, y)
    logistic_prob = scaled_log_reg.predict_proba(X)[:, 1]
    logistic_pred = (logistic_prob >= best_logistic_threshold).astype(int)

    # Fit preferred Random Forest on full rural data
    print("Fitting Random Forest on full rural data...")
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X, y)
    rf_prob = rf.predict_proba(X)[:, 1]
    rf_pred = (rf_prob >= 0.50).astype(int)

    # Build prediction output table
    predictions = readable_df.copy()
    predictions["scaled_logistic_probability"] = logistic_prob
    predictions["scaled_logistic_predicted_class"] = logistic_pred

    predictions["random_forest_probability"] = rf_prob
    predictions["random_forest_predicted_class"] = rf_pred

    if preferred_model_name == "random_forest":
        predictions["preferred_model_name"] = "random_forest"
        predictions["preferred_model_probability"] = predictions["random_forest_probability"]
        predictions["preferred_model_predicted_class"] = predictions["random_forest_predicted_class"]
    else:
        predictions["preferred_model_name"] = "scaled_logistic_regression"
        predictions["preferred_model_probability"] = predictions["scaled_logistic_probability"]
        predictions["preferred_model_predicted_class"] = predictions["scaled_logistic_predicted_class"]

    predictions["preferred_risk_band"] = assign_probability_band(
        predictions["preferred_model_probability"]
    )

    predictions = predictions.sort_values(
        "preferred_model_probability",
        ascending=False,
    ).reset_index(drop=True)

    predictions["preferred_probability_rank"] = predictions.index + 1

    # Build top-risk shortlist
    high_risk_zones = predictions[predictions["preferred_risk_band"] == "high"].copy()
    high_risk_zones = high_risk_zones.sort_values(
        "preferred_model_probability",
        ascending=False,
    ).reset_index(drop=True)

    top_50_rural = high_risk_zones.head(50).copy()

    # Build map-ready spatial layer
    zone_cols = [
        "dz_code_2022",
        "dz_name_2022",
        "ur6_name",
        "ur8_name",
        "is_rural",
        "is_accessible_rural",
        "is_remote_rural",
        "geometry",
    ]
    zones_map = zones_master_2022[zone_cols].copy()

    map_ready = zones_map.merge(
        predictions.drop(
            columns=[
                "dz_name_2022",
                "ur6_name",
                "ur8_name",
                "is_rural",
                "is_accessible_rural",
                "is_remote_rural",
            ],
            errors="ignore",
        ),
        on="dz_code_2022",
        how="inner",
        validate="1:1",
    )

    # Save outputs
    predictions_csv = output_dir / "rural_zone_prediction_outputs.csv"
    top50_csv = output_dir / "top_50_high_risk_rural_zones.csv"
    map_ready_gpkg = output_dir / "rural_zone_prediction_outputs.gpkg"
    map_ready_csv = output_dir / "rural_zone_prediction_outputs_map_ready.csv"

    predictions.to_csv(predictions_csv, index=False)
    top_50_rural.to_csv(top50_csv, index=False)

    map_ready.to_file(
        map_ready_gpkg,
        layer="rural_zone_prediction_outputs",
        driver="GPKG",
    )
    pd.DataFrame(map_ready.drop(columns="geometry")).to_csv(map_ready_csv, index=False)

    print("\n--- Prediction Output Summary ---")
    print(f"Prediction rows: {len(predictions)}")
    print(f"High-risk rural rows: {len(high_risk_zones)}")
    print(f"Top 50 shortlist rows: {len(top_50_rural)}")

    print("\nPreferred risk band counts:")
    print(predictions["preferred_risk_band"].value_counts(dropna=False))

    print("\nPreferred predicted class counts:")
    print(predictions["preferred_model_predicted_class"].value_counts(dropna=False))

    print("\nTop 10 rural zones by preferred probability:")
    print(
        predictions[
            [
                "dz_code_2022",
                "dz_name_2022",
                "ur6_name",
                "ur8_name",
                "preferred_model_probability",
                "preferred_model_predicted_class",
                "preferred_risk_band",
                "underserved_baseline",
                "critical_underserved_baseline",
            ]
        ].head(10)
    )

    print("\nbuild_prediction_outputs completed successfully.")
    print(f"Prediction outputs CSV: {predictions_csv}")
    print(f"Top 50 shortlist CSV: {top50_csv}")
    print(f"Map-ready GPKG: {map_ready_gpkg}")
    print(f"Map-ready CSV: {map_ready_csv}")


if __name__ == "__main__":
    main()