import json
import pandas as pd
import geopandas as gpd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from config.paths import PROCESSED_DIR


TARGET_COL = "bank_access_deterioration_flag"
RANDOM_STATE = 42
DEFAULT_FINAL_THRESHOLD = 0.50
DEFAULT_POLICY_THRESHOLD = 0.65


def load_temporal_model_choice(results_dir):
    summary_path = results_dir / "temporal_model_summary.json"

    preferred_model_name = "scaled_logistic_regression"

    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        preferred_model_name = payload.get("preferred_model", {}).get(
            "model_name",
            "scaled_logistic_regression",
        )

    return preferred_model_name


def load_temporal_thresholds(results_dir):
    """
    Load validated thresholds from temporal_scaled_logistic_validation_summary.json.
    Falls back to sensible defaults if file is missing.
    """
    summary_path = results_dir / "temporal_scaled_logistic_validation_summary.json"

    final_threshold = DEFAULT_FINAL_THRESHOLD
    policy_threshold = DEFAULT_POLICY_THRESHOLD

    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        final_threshold = payload.get("best_threshold_by_f1", {}).get(
            "threshold",
            DEFAULT_FINAL_THRESHOLD,
        )

        policy_threshold = payload.get("closest_threshold_to_actual_positive_rate", {}).get(
            "threshold",
            DEFAULT_POLICY_THRESHOLD,
        )

    return float(final_threshold), float(policy_threshold)


def assign_probability_band(prob_series: pd.Series) -> pd.Series:
    ranked = prob_series.rank(method="first")
    bands = pd.qcut(
        ranked,
        q=3,
        labels=["low", "medium", "high"],
    )
    return bands.astype(str)


def main() -> None:
    print("Starting build_temporal_prediction_outputs...")

    ml_dir = PROCESSED_DIR / "ml"
    results_dir = ml_dir / "temporal_model_results"
    output_dir = ml_dir / "temporal_prediction_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    encoded_path = ml_dir / "model_ready_temporal_rural_encoded.csv"
    readable_path = ml_dir / "model_ready_temporal_rural_readable.csv"
    geography_path = PROCESSED_DIR / "geography" / "zones_master_2022.gpkg"

    print(f"Loading rural temporal encoded table from: {encoded_path}")
    encoded_df = pd.read_csv(encoded_path)

    print(f"Loading rural temporal readable table from: {readable_path}")
    readable_df = pd.read_csv(readable_path)

    print(f"Loading geography from: {geography_path}")
    zones_master_2022 = gpd.read_file(geography_path, layer="zones_master_2022")

    preferred_model_name = load_temporal_model_choice(results_dir)
    final_threshold, policy_threshold = load_temporal_thresholds(results_dir)

    print(f"Preferred temporal model from validation: {preferred_model_name}")
    print(f"Validated final threshold (best F1): {final_threshold}")
    print(f"Validated policy shortlist threshold: {policy_threshold}")

    if TARGET_COL not in encoded_df.columns:
        raise ValueError(f"Missing target column in encoded temporal table: {TARGET_COL}")

    aux_target_cols = [
        col for col in [
            "bank_access_deterioration_flag",
            "bank_access_major_deterioration_flag",
            "bank_access_severe_deterioration_flag",
        ]
        if col in encoded_df.columns
    ]

    X = encoded_df.drop(columns=aux_target_cols).copy()
    y = encoded_df[TARGET_COL].copy()

    print(f"\nEncoded rural rows: {len(encoded_df)}")
    print(f"Readable rural rows: {len(readable_df)}")
    print(f"Feature columns: {len(X.columns)}")

    print("\nFitting scaled Logistic Regression on full rural temporal data...")
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
    logistic_pred_final = (logistic_prob >= final_threshold).astype(int)
    logistic_pred_policy = (logistic_prob >= policy_threshold).astype(int)

    print("Fitting Random Forest on full rural temporal data...")
    rf = RandomForestClassifier(
        n_estimators=400,
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

    predictions = readable_df.copy()

    predictions["scaled_logistic_probability"] = logistic_prob
    predictions["scaled_logistic_predicted_class_final"] = logistic_pred_final
    predictions["scaled_logistic_predicted_class_policy"] = logistic_pred_policy

    predictions["random_forest_probability"] = rf_prob
    predictions["random_forest_predicted_class"] = rf_pred

    if preferred_model_name == "random_forest":
        predictions["preferred_temporal_model_name"] = "random_forest"
        predictions["preferred_temporal_probability"] = predictions["random_forest_probability"]
        predictions["preferred_temporal_predicted_class_final"] = predictions["random_forest_predicted_class"]
        predictions["preferred_temporal_predicted_class_policy"] = predictions["random_forest_predicted_class"]
    else:
        predictions["preferred_temporal_model_name"] = "scaled_logistic_regression"
        predictions["preferred_temporal_probability"] = predictions["scaled_logistic_probability"]
        predictions["preferred_temporal_predicted_class_final"] = predictions["scaled_logistic_predicted_class_final"]
        predictions["preferred_temporal_predicted_class_policy"] = predictions["scaled_logistic_predicted_class_policy"]

    predictions["temporal_risk_band"] = assign_probability_band(
        predictions["preferred_temporal_probability"]
    )

    predictions = predictions.sort_values(
        by=["preferred_temporal_probability", "post_covid_closures_total", "closure_rate_change_post_minus_pre"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    predictions["temporal_probability_rank"] = predictions.index + 1

    high_risk = predictions[predictions["temporal_risk_band"] == "high"].copy()
    top_100 = high_risk.head(100).copy()
    top_50 = high_risk.head(50).copy()

    policy_shortlist = predictions[
        predictions["preferred_temporal_predicted_class_policy"] == 1
    ].copy()

    policy_top_100 = policy_shortlist.head(100).copy()
    policy_top_50 = policy_shortlist.head(50).copy()

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
    zone_cols = [col for col in zone_cols if col in zones_master_2022.columns]
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

    summary_by_ur6 = (
        predictions.groupby("ur6_name", dropna=False)
        .agg(
            rural_zone_count=("dz_code_2022", "count"),
            avg_temporal_probability=("preferred_temporal_probability", "mean"),
            predicted_deterioration_count_final=("preferred_temporal_predicted_class_final", "sum"),
            predicted_deterioration_count_policy=("preferred_temporal_predicted_class_policy", "sum"),
            actual_deterioration_count=("bank_access_deterioration_flag", "sum"),
            major_deterioration_count=("bank_access_major_deterioration_flag", "sum"),
            severe_deterioration_count=("bank_access_severe_deterioration_flag", "sum"),
        )
        .reset_index()
        .sort_values("avg_temporal_probability", ascending=False)
        .reset_index(drop=True)
    )

    summary_by_ur6["predicted_deterioration_share_final"] = (
        summary_by_ur6["predicted_deterioration_count_final"] / summary_by_ur6["rural_zone_count"]
    )
    summary_by_ur6["predicted_deterioration_share_policy"] = (
        summary_by_ur6["predicted_deterioration_count_policy"] / summary_by_ur6["rural_zone_count"]
    )
    summary_by_ur6["actual_deterioration_share"] = (
        summary_by_ur6["actual_deterioration_count"] / summary_by_ur6["rural_zone_count"]
    )

    class_band_summary = pd.DataFrame(
        {
            "metric": [
                "predicted_positive_count_final",
                "predicted_negative_count_final",
                "predicted_positive_count_policy",
                "predicted_negative_count_policy",
                "high_risk_band_count",
                "medium_risk_band_count",
                "low_risk_band_count",
                "actual_deterioration_count",
                "actual_major_deterioration_count",
                "actual_severe_deterioration_count",
                "final_threshold_used",
                "policy_threshold_used",
            ],
            "value": [
                int((predictions["preferred_temporal_predicted_class_final"] == 1).sum()),
                int((predictions["preferred_temporal_predicted_class_final"] == 0).sum()),
                int((predictions["preferred_temporal_predicted_class_policy"] == 1).sum()),
                int((predictions["preferred_temporal_predicted_class_policy"] == 0).sum()),
                int((predictions["temporal_risk_band"] == "high").sum()),
                int((predictions["temporal_risk_band"] == "medium").sum()),
                int((predictions["temporal_risk_band"] == "low").sum()),
                int(predictions["bank_access_deterioration_flag"].sum()),
                int(predictions["bank_access_major_deterioration_flag"].sum()),
                int(predictions["bank_access_severe_deterioration_flag"].sum()),
                float(final_threshold),
                float(policy_threshold),
            ],
        }
    )

    predictions_csv = output_dir / "temporal_rural_prediction_outputs.csv"
    summary_ur6_csv = output_dir / "temporal_prediction_summary_by_ur6.csv"
    class_band_csv = output_dir / "temporal_prediction_class_band_summary.csv"
    top100_csv = output_dir / "temporal_top_100_high_risk_rural_zones.csv"
    top50_csv = output_dir / "temporal_top_50_high_risk_rural_zones.csv"
    policy_top100_csv = output_dir / "temporal_policy_top_100_rural_zones.csv"
    policy_top50_csv = output_dir / "temporal_policy_top_50_rural_zones.csv"
    map_ready_gpkg = output_dir / "temporal_rural_prediction_outputs.gpkg"
    map_ready_csv = output_dir / "temporal_rural_prediction_outputs_map_ready.csv"

    predictions.to_csv(predictions_csv, index=False)
    summary_by_ur6.to_csv(summary_ur6_csv, index=False)
    class_band_summary.to_csv(class_band_csv, index=False)
    top_100.to_csv(top100_csv, index=False)
    top_50.to_csv(top50_csv, index=False)
    policy_top_100.to_csv(policy_top100_csv, index=False)
    policy_top_50.to_csv(policy_top50_csv, index=False)

    map_ready.to_file(
        map_ready_gpkg,
        layer="temporal_rural_prediction_outputs",
        driver="GPKG",
    )
    pd.DataFrame(map_ready.drop(columns="geometry")).to_csv(map_ready_csv, index=False)

    print("\n--- Temporal Prediction Output Summary ---")
    print(f"Prediction rows: {len(predictions)}")
    print(f"High-risk rows: {len(high_risk)}")
    print(f"Top 100 high-risk rows: {len(top_100)}")
    print(f"Top 50 high-risk rows: {len(top_50)}")
    print(f"Policy shortlist rows: {len(policy_shortlist)}")
    print(f"Policy top 100 rows: {len(policy_top_100)}")
    print(f"Policy top 50 rows: {len(policy_top_50)}")

    print("\nTemporal risk band counts:")
    print(predictions["temporal_risk_band"].value_counts(dropna=False))

    print("\nPreferred temporal predicted class counts (final threshold):")
    print(predictions["preferred_temporal_predicted_class_final"].value_counts(dropna=False))

    print("\nPreferred temporal predicted class counts (policy threshold):")
    print(predictions["preferred_temporal_predicted_class_policy"].value_counts(dropna=False))

    print("\nSummary by UR6:")
    print(summary_by_ur6)

    print("\nTop 10 temporal high-risk rural zones:")
    top_cols = [
        "dz_code_2022",
        "dz_name_2022",
        "ur6_name",
        "ur8_name",
        "preferred_temporal_probability",
        "preferred_temporal_predicted_class_final",
        "preferred_temporal_predicted_class_policy",
        "temporal_risk_band",
        "bank_access_deterioration_flag",
        "bank_access_major_deterioration_flag",
        "bank_access_severe_deterioration_flag",
        "post_covid_closures_total",
        "closure_rate_change_post_minus_pre",
    ]
    top_cols = [col for col in top_cols if col in predictions.columns]
    print(predictions[top_cols].head(10))

    print("\nbuild_temporal_prediction_outputs completed successfully.")
    print(f"Prediction outputs CSV: {predictions_csv}")
    print(f"Summary by UR6 CSV: {summary_ur6_csv}")
    print(f"Class/band summary CSV: {class_band_csv}")
    print(f"Top 100 high-risk CSV: {top100_csv}")
    print(f"Top 50 high-risk CSV: {top50_csv}")
    print(f"Policy top 100 CSV: {policy_top100_csv}")
    print(f"Policy top 50 CSV: {policy_top50_csv}")
    print(f"Map-ready GPKG: {map_ready_gpkg}")
    print(f"Map-ready CSV: {map_ready_csv}")


if __name__ == "__main__":
    main()