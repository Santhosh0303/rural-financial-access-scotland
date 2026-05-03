import json
from pathlib import Path
import numpy as np
import pandas as pd

from config.paths import PROCESSED_DIR


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV file: {path}")
    return pd.read_csv(path)


def get_metric_value(summary_df: pd.DataFrame, metric_name: str):
    row = summary_df.loc[summary_df["metric"] == metric_name, "value"]
    if row.empty:
        return np.nan
    return row.iloc[0]


def build_overview_comparison(
    service_points: pd.DataFrame,
    v1_access: pd.DataFrame,
    v1_predictions: pd.DataFrame,
    v1_interventions: pd.DataFrame,
    v1_simulations: pd.DataFrame,
    v1_model_summary: dict,
    v2_temporal_predictions: pd.DataFrame,
    v2_temporal_overall_summary: pd.DataFrame,
    v2_model_summary: dict,
    v2_threshold_summary: dict,
) -> pd.DataFrame:
    v1_preferred_model = v1_model_summary.get("preferred_model", {}).get("model_name", "unknown")
    v2_preferred_model = v2_model_summary.get("preferred_model", {}).get("model_name", "unknown")

    v2_best_f1_threshold = v2_threshold_summary.get("best_threshold_by_f1", {}).get("threshold", np.nan)
    v2_policy_threshold = v2_threshold_summary.get("closest_threshold_to_actual_positive_rate", {}).get("threshold", np.nan)

    rows = [
        {
            "metric_group": "coverage",
            "metric": "total_zones",
            "v1_value": int(v1_access["dz_code_2022"].nunique()),
            "v2_value": int(get_metric_value(v2_temporal_overall_summary, "total_zones")),
        },
        {
            "metric_group": "coverage",
            "metric": "rural_zones",
            "v1_value": int(len(v1_predictions)),
            "v2_value": int(get_metric_value(v2_temporal_overall_summary, "rural_zones")),
        },
        {
            "metric_group": "services",
            "metric": "current_service_points_total",
            "v1_value": int(len(service_points)),
            "v2_value": np.nan,
        },
        {
            "metric_group": "services",
            "metric": "mean_any_access_point_km_current",
            "v1_value": float(v1_access["dist_to_nearest_any_access_point_km"].mean()),
            "v2_value": np.nan,
        },
        {
            "metric_group": "v1_underserved",
            "metric": "baseline_underserved_rural_count",
            "v1_value": int(v1_predictions["underserved_baseline"].sum()),
            "v2_value": np.nan,
        },
        {
            "metric_group": "v1_underserved",
            "metric": "critical_underserved_rural_count",
            "v1_value": int(v1_predictions["critical_underserved_baseline"].sum()),
            "v2_value": np.nan,
        },
        {
            "metric_group": "v1_model",
            "metric": "v1_preferred_model",
            "v1_value": v1_preferred_model,
            "v2_value": np.nan,
        },
        {
            "metric_group": "v1_model",
            "metric": "v1_predicted_positive_count",
            "v1_value": int(v1_predictions["preferred_model_predicted_class"].sum()),
            "v2_value": np.nan,
        },
        {
            "metric_group": "v1_model",
            "metric": "v1_high_risk_band_count",
            "v1_value": int((v1_predictions["preferred_risk_band"] == "high").sum()),
            "v2_value": np.nan,
        },
        {
            "metric_group": "v1_policy",
            "metric": "v1_intervention_candidate_count",
            "v1_value": int(len(v1_interventions)),
            "v2_value": np.nan,
        },
        {
            "metric_group": "v1_policy",
            "metric": "v1_simulation_row_count",
            "v1_value": int(len(v1_simulations)),
            "v2_value": np.nan,
        },
        {
            "metric_group": "v2_temporal_access",
            "metric": "mean_bank_km_2019",
            "v1_value": np.nan,
            "v2_value": float(get_metric_value(v2_temporal_overall_summary, "mean_bank_km_2019")),
        },
        {
            "metric_group": "v2_temporal_access",
            "metric": "mean_bank_km_2023",
            "v1_value": np.nan,
            "v2_value": float(get_metric_value(v2_temporal_overall_summary, "mean_bank_km_2023")),
        },
        {
            "metric_group": "v2_temporal_access",
            "metric": "zones_with_bank_access_deterioration",
            "v1_value": np.nan,
            "v2_value": int(get_metric_value(v2_temporal_overall_summary, "zones_with_bank_access_deterioration")),
        },
        {
            "metric_group": "v2_temporal_access",
            "metric": "zones_with_major_bank_access_deterioration",
            "v1_value": np.nan,
            "v2_value": int(get_metric_value(v2_temporal_overall_summary, "zones_with_major_bank_access_deterioration")),
        },
        {
            "metric_group": "v2_temporal_access",
            "metric": "zones_with_severe_bank_access_deterioration",
            "v1_value": np.nan,
            "v2_value": int(get_metric_value(v2_temporal_overall_summary, "zones_with_severe_bank_access_deterioration")),
        },
        {
            "metric_group": "v2_model",
            "metric": "v2_preferred_model",
            "v1_value": np.nan,
            "v2_value": v2_preferred_model,
        },
        {
            "metric_group": "v2_model",
            "metric": "v2_best_f1_threshold",
            "v1_value": np.nan,
            "v2_value": float(v2_best_f1_threshold),
        },
        {
            "metric_group": "v2_model",
            "metric": "v2_policy_threshold",
            "v1_value": np.nan,
            "v2_value": float(v2_policy_threshold),
        },
        {
            "metric_group": "v2_model",
            "metric": "v2_predicted_positive_count_final",
            "v1_value": np.nan,
            "v2_value": int(v2_temporal_predictions["preferred_temporal_predicted_class_final"].sum()),
        },
        {
            "metric_group": "v2_model",
            "metric": "v2_predicted_positive_count_policy",
            "v1_value": np.nan,
            "v2_value": int(v2_temporal_predictions["preferred_temporal_predicted_class_policy"].sum()),
        },
        {
            "metric_group": "v2_model",
            "metric": "v2_actual_deterioration_count",
            "v1_value": np.nan,
            "v2_value": int(v2_temporal_predictions["bank_access_deterioration_flag"].sum()),
        },
        {
            "metric_group": "v2_model",
            "metric": "v2_high_risk_band_count",
            "v1_value": np.nan,
            "v2_value": int((v2_temporal_predictions["temporal_risk_band"] == "high").sum()),
        },
    ]

    return pd.DataFrame(rows)


def build_model_comparison(
    v1_cv_summary: pd.DataFrame,
    v2_cv_summary: pd.DataFrame,
    v1_model_summary: dict,
    v2_model_summary: dict,
) -> pd.DataFrame:
    v1_preferred = v1_model_summary.get("preferred_model", {}).get("model_name", "unknown")
    v2_preferred = v2_model_summary.get("preferred_model", {}).get("model_name", "unknown")

    v1 = v1_cv_summary.copy()
    v1["version"] = "v1_current_baseline"
    v1["preferred_model_flag"] = (v1["model_name"] == v1_preferred).astype(int)

    v2 = v2_cv_summary.copy()
    v2["version"] = "v2_temporal"
    v2["preferred_model_flag"] = (v2["model_name"] == v2_preferred).astype(int)

    cols = [
        "version",
        "model_name",
        "preferred_model_flag",
        "accuracy_mean",
        "accuracy_std",
        "precision_mean",
        "precision_std",
        "recall_mean",
        "recall_std",
        "f1_mean",
        "f1_std",
        "roc_auc_mean",
        "roc_auc_std",
    ]
    cols = [col for col in cols if col in v1.columns or col in v2.columns]

    return pd.concat([v1[cols], v2[cols]], ignore_index=True)


def build_ur6_comparison(
    v1_predictions: pd.DataFrame,
    v2_predictions: pd.DataFrame,
) -> pd.DataFrame:
    v1_summary = (
        v1_predictions.groupby("ur6_name", dropna=False)
        .agg(
            v1_rural_zone_count=("dz_code_2022", "count"),
            v1_avg_probability=("preferred_model_probability", "mean"),
            v1_predicted_positive_count=("preferred_model_predicted_class", "sum"),
            v1_baseline_underserved_count=("underserved_baseline", "sum"),
            v1_critical_underserved_count=("critical_underserved_baseline", "sum"),
        )
        .reset_index()
    )
    v1_summary["v1_predicted_positive_share"] = (
        v1_summary["v1_predicted_positive_count"] / v1_summary["v1_rural_zone_count"]
    )
    v1_summary["v1_baseline_underserved_share"] = (
        v1_summary["v1_baseline_underserved_count"] / v1_summary["v1_rural_zone_count"]
    )

    v2_summary = (
        v2_predictions.groupby("ur6_name", dropna=False)
        .agg(
            v2_rural_zone_count=("dz_code_2022", "count"),
            v2_avg_probability=("preferred_temporal_probability", "mean"),
            v2_predicted_positive_count_final=("preferred_temporal_predicted_class_final", "sum"),
            v2_predicted_positive_count_policy=("preferred_temporal_predicted_class_policy", "sum"),
            v2_actual_deterioration_count=("bank_access_deterioration_flag", "sum"),
            v2_major_deterioration_count=("bank_access_major_deterioration_flag", "sum"),
            v2_severe_deterioration_count=("bank_access_severe_deterioration_flag", "sum"),
        )
        .reset_index()
    )
    v2_summary["v2_predicted_positive_share_final"] = (
        v2_summary["v2_predicted_positive_count_final"] / v2_summary["v2_rural_zone_count"]
    )
    v2_summary["v2_predicted_positive_share_policy"] = (
        v2_summary["v2_predicted_positive_count_policy"] / v2_summary["v2_rural_zone_count"]
    )
    v2_summary["v2_actual_deterioration_share"] = (
        v2_summary["v2_actual_deterioration_count"] / v2_summary["v2_rural_zone_count"]
    )

    comparison = v1_summary.merge(
        v2_summary,
        on="ur6_name",
        how="outer",
        validate="1:1",
    ).sort_values("ur6_name").reset_index(drop=True)

    return comparison


def build_shortlist_comparison(
    v1_predictions: pd.DataFrame,
    v2_predictions: pd.DataFrame,
) -> pd.DataFrame:
    v1_shortlist = (
        v1_predictions.copy()
        .sort_values("preferred_model_probability", ascending=False)
        .reset_index(drop=True)
        .head(50)
    )
    v1_shortlist["rank"] = v1_shortlist.index + 1
    v1_shortlist["version"] = "v1_current_baseline"
    v1_shortlist["selection_basis"] = "top_50_high_risk_current"

    v1_shortlist = v1_shortlist[
        [
            "version",
            "selection_basis",
            "rank",
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
    ].rename(
        columns={
            "preferred_model_probability": "score_probability",
            "preferred_model_predicted_class": "predicted_class",
            "preferred_risk_band": "risk_band",
            "underserved_baseline": "actual_flag_1",
            "critical_underserved_baseline": "actual_flag_2",
        }
    )

    v2_shortlist = (
        v2_predictions[v2_predictions["preferred_temporal_predicted_class_policy"] == 1]
        .copy()
        .sort_values("preferred_temporal_probability", ascending=False)
        .reset_index(drop=True)
        .head(50)
    )
    v2_shortlist["rank"] = v2_shortlist.index + 1
    v2_shortlist["version"] = "v2_temporal"
    v2_shortlist["selection_basis"] = "top_50_policy_temporal"

    v2_shortlist = v2_shortlist[
        [
            "version",
            "selection_basis",
            "rank",
            "dz_code_2022",
            "dz_name_2022",
            "ur6_name",
            "ur8_name",
            "preferred_temporal_probability",
            "preferred_temporal_predicted_class_policy",
            "temporal_risk_band",
            "bank_access_deterioration_flag",
            "bank_access_severe_deterioration_flag",
        ]
    ].rename(
        columns={
            "preferred_temporal_probability": "score_probability",
            "preferred_temporal_predicted_class_policy": "predicted_class",
            "temporal_risk_band": "risk_band",
            "bank_access_deterioration_flag": "actual_flag_1",
            "bank_access_severe_deterioration_flag": "actual_flag_2",
        }
    )

    return pd.concat([v1_shortlist, v2_shortlist], ignore_index=True)


def build_threshold_comparison(
    v2_threshold_summary: dict,
) -> pd.DataFrame:
    rows = []

    for label, key in [
        ("best_threshold_by_f1", "best_threshold_by_f1"),
        ("best_threshold_by_recall", "best_threshold_by_recall"),
        ("closest_threshold_to_actual_positive_rate", "closest_threshold_to_actual_positive_rate"),
    ]:
        payload = v2_threshold_summary.get(key, {})
        rows.append(
            {
                "threshold_type": label,
                "threshold": payload.get("threshold", np.nan),
                "accuracy": payload.get("accuracy", np.nan),
                "precision": payload.get("precision", np.nan),
                "recall": payload.get("recall", np.nan),
                "f1": payload.get("f1", np.nan),
                "predicted_positive_count": payload.get("predicted_positive_count", np.nan),
                "predicted_positive_rate": payload.get("predicted_positive_rate", np.nan),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    print("Starting V1 vs V2 comparison export build...")

    processed_dir = PROCESSED_DIR
    comparison_dir = processed_dir / "comparison_exports"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    service_points_path = processed_dir / "services_current" / "service_points_current.csv"
    v1_access_path = processed_dir / "accessibility" / "zone_accessibility_baseline_2022.csv"
    v1_predictions_path = processed_dir / "ml" / "prediction_outputs" / "rural_zone_prediction_outputs.csv"
    v1_interventions_path = processed_dir / "scenario" / "scenario_interventions_all.csv"
    v1_simulations_path = processed_dir / "scenario" / "scenario_simulation_baseline_all.csv"
    v1_cv_summary_path = processed_dir / "ml" / "refined_model_results" / "refined_model_cv_summary.csv"
    v1_model_summary_path = processed_dir / "ml" / "refined_model_results" / "refined_model_summary.json"

    v2_temporal_predictions_path = processed_dir / "ml" / "temporal_prediction_outputs" / "temporal_rural_prediction_outputs.csv"
    v2_temporal_overall_summary_path = processed_dir / "accessibility" / "bank_accessibility_temporal_overall_summary.csv"
    v2_temporal_cv_summary_path = processed_dir / "ml" / "temporal_model_results" / "temporal_model_cv_summary.csv"
    v2_model_summary_path = processed_dir / "ml" / "temporal_model_results" / "temporal_model_summary.json"
    v2_threshold_summary_path = processed_dir / "ml" / "temporal_model_results" / "temporal_scaled_logistic_validation_summary.json"

    print(f"Loading V1 service points from: {service_points_path}")
    service_points = load_csv(service_points_path)

    print(f"Loading V1 accessibility from: {v1_access_path}")
    v1_access = load_csv(v1_access_path)

    print(f"Loading V1 predictions from: {v1_predictions_path}")
    v1_predictions = load_csv(v1_predictions_path)

    print(f"Loading V1 interventions from: {v1_interventions_path}")
    v1_interventions = load_csv(v1_interventions_path)

    print(f"Loading V1 simulations from: {v1_simulations_path}")
    v1_simulations = load_csv(v1_simulations_path)

    print(f"Loading V1 CV summary from: {v1_cv_summary_path}")
    v1_cv_summary = load_csv(v1_cv_summary_path)

    print(f"Loading V1 model summary from: {v1_model_summary_path}")
    v1_model_summary = load_json(v1_model_summary_path)

    print(f"Loading V2 temporal predictions from: {v2_temporal_predictions_path}")
    v2_temporal_predictions = load_csv(v2_temporal_predictions_path)

    print(f"Loading V2 temporal overall summary from: {v2_temporal_overall_summary_path}")
    v2_temporal_overall_summary = load_csv(v2_temporal_overall_summary_path)

    print(f"Loading V2 temporal CV summary from: {v2_temporal_cv_summary_path}")
    v2_temporal_cv_summary = load_csv(v2_temporal_cv_summary_path)

    print(f"Loading V2 model summary from: {v2_model_summary_path}")
    v2_model_summary = load_json(v2_model_summary_path)

    print(f"Loading V2 threshold summary from: {v2_threshold_summary_path}")
    v2_threshold_summary = load_json(v2_threshold_summary_path)

    print("\n--- Input Row Summary ---")
    print(f"Service points rows: {len(service_points)}")
    print(f"V1 access rows: {len(v1_access)}")
    print(f"V1 prediction rows: {len(v1_predictions)}")
    print(f"V1 intervention rows: {len(v1_interventions)}")
    print(f"V1 simulation rows: {len(v1_simulations)}")
    print(f"V1 CV summary rows: {len(v1_cv_summary)}")
    print(f"V2 temporal prediction rows: {len(v2_temporal_predictions)}")
    print(f"V2 temporal overall summary rows: {len(v2_temporal_overall_summary)}")
    print(f"V2 temporal CV summary rows: {len(v2_temporal_cv_summary)}")

    overview_comparison = build_overview_comparison(
        service_points=service_points,
        v1_access=v1_access,
        v1_predictions=v1_predictions,
        v1_interventions=v1_interventions,
        v1_simulations=v1_simulations,
        v1_model_summary=v1_model_summary,
        v2_temporal_predictions=v2_temporal_predictions,
        v2_temporal_overall_summary=v2_temporal_overall_summary,
        v2_model_summary=v2_model_summary,
        v2_threshold_summary=v2_threshold_summary,
    )

    model_comparison = build_model_comparison(
        v1_cv_summary=v1_cv_summary,
        v2_cv_summary=v2_temporal_cv_summary,
        v1_model_summary=v1_model_summary,
        v2_model_summary=v2_model_summary,
    )

    ur6_comparison = build_ur6_comparison(
        v1_predictions=v1_predictions,
        v2_predictions=v2_temporal_predictions,
    )

    shortlist_comparison = build_shortlist_comparison(
        v1_predictions=v1_predictions,
        v2_predictions=v2_temporal_predictions,
    )

    threshold_comparison = build_threshold_comparison(
        v2_threshold_summary=v2_threshold_summary,
    )

    overview_path = comparison_dir / "v1_v2_overview_comparison.csv"
    model_path = comparison_dir / "v1_v2_model_comparison.csv"
    ur6_path = comparison_dir / "v1_v2_ur6_comparison.csv"
    shortlist_path = comparison_dir / "v1_v2_top50_shortlist_comparison.csv"
    threshold_path = comparison_dir / "v2_threshold_comparison.csv"

    overview_comparison.to_csv(overview_path, index=False)
    model_comparison.to_csv(model_path, index=False)
    ur6_comparison.to_csv(ur6_path, index=False)
    shortlist_comparison.to_csv(shortlist_path, index=False)
    threshold_comparison.to_csv(threshold_path, index=False)

    print("\n--- Comparison Export Summary ---")
    print(f"Overview comparison rows: {len(overview_comparison)}")
    print(f"Model comparison rows: {len(model_comparison)}")
    print(f"UR6 comparison rows: {len(ur6_comparison)}")
    print(f"Top-50 shortlist comparison rows: {len(shortlist_comparison)}")
    print(f"Threshold comparison rows: {len(threshold_comparison)}")

    print("\nPreview of overview comparison:")
    print(overview_comparison.head(15))

    print("\nPreview of model comparison:")
    print(model_comparison)

    print("\nPreview of UR6 comparison:")
    print(ur6_comparison)

    print("\nPreview of threshold comparison:")
    print(threshold_comparison)

    print("\nV1 vs V2 comparison export build completed successfully.")
    print(f"Overview comparison CSV: {overview_path}")
    print(f"Model comparison CSV: {model_path}")
    print(f"UR6 comparison CSV: {ur6_path}")
    print(f"Top-50 shortlist comparison CSV: {shortlist_path}")
    print(f"Threshold comparison CSV: {threshold_path}")


if __name__ == "__main__":
    main()