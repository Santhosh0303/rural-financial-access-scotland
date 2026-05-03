import json
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd

from config.paths import PROCESSED_DIR


OUTPUT_DIR = PROCESSED_DIR / "dashboard_exports" / "dashboard3_enhanced"


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required CSV file: {path}")
    return pd.read_csv(path)


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing required JSON file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def first_existing_col(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    if required:
        raise ValueError(f"None of these expected columns were found: {candidates}")
    return None


def read_zones_with_coordinates(zones_path: Path) -> pd.DataFrame:
    zones = gpd.read_file(zones_path, layer="zones_master_2022")

    # Representative points are safer than centroids because they stay inside polygons.
    zones_points = zones.copy()
    zones_points["geometry"] = zones_points.geometry.representative_point()
    zones_points = zones_points.to_crs("EPSG:4326")
    zones_points["longitude"] = zones_points.geometry.x
    zones_points["latitude"] = zones_points.geometry.y

    keep_cols = [
        "dz_code_2022",
        "dz_name_2022",
        "ur6_name",
        "ur8_name",
        "is_rural",
        "is_accessible_rural",
        "is_remote_rural",
        "longitude",
        "latitude",
    ]
    keep_cols = [col for col in keep_cols if col in zones_points.columns]

    return zones_points[keep_cols].drop_duplicates(subset=["dz_code_2022"]).copy()


def build_model_kpis(
    predictions: pd.DataFrame,
    cv_summary: pd.DataFrame,
    model_summary: dict,
) -> pd.DataFrame:
    preferred_model = model_summary.get("preferred_model", {}).get("model_name", "unknown")

    preferred_row = cv_summary[cv_summary["model_name"] == preferred_model].copy()
    if preferred_row.empty:
        preferred_row = cv_summary.sort_values(
            by=["f1_mean", "roc_auc_mean"],
            ascending=False,
        ).head(1)

    preferred_row = preferred_row.iloc[0]

    prob_col = first_existing_col(
        predictions,
        ["preferred_model_probability", "preferred_probability", "random_forest_probability"],
    )
    pred_col = first_existing_col(
        predictions,
        ["preferred_model_predicted_class", "preferred_predicted_class"],
    )
    target_col = first_existing_col(predictions, ["underserved_baseline"])

    critical_col = first_existing_col(
        predictions,
        ["critical_underserved_baseline"],
        required=False,
    )

    risk_band_col = first_existing_col(
        predictions,
        ["preferred_risk_band", "risk_band"],
        required=False,
    )

    actual_positive = int(predictions[target_col].sum())
    predicted_positive = int(predictions[pred_col].sum())

    high_risk_count = (
        int((predictions[risk_band_col].astype(str).str.lower() == "high").sum())
        if risk_band_col
        else np.nan
    )

    rows = [
        {
            "kpi_group": "model_performance",
            "metric": "preferred_model",
            "value": preferred_model,
            "display_value": preferred_model,
            "description": "Preferred Version 1 model selected from cross-validation.",
        },
        {
            "kpi_group": "model_performance",
            "metric": "cv_accuracy_mean",
            "value": float(preferred_row.get("accuracy_mean", np.nan)),
            "display_value": f"{float(preferred_row.get('accuracy_mean', np.nan)):.3f}",
            "description": "Mean cross-validation accuracy for the preferred model.",
        },
        {
            "kpi_group": "model_performance",
            "metric": "cv_precision_mean",
            "value": float(preferred_row.get("precision_mean", np.nan)),
            "display_value": f"{float(preferred_row.get('precision_mean', np.nan)):.3f}",
            "description": "Mean cross-validation precision for the preferred model.",
        },
        {
            "kpi_group": "model_performance",
            "metric": "cv_recall_mean",
            "value": float(preferred_row.get("recall_mean", np.nan)),
            "display_value": f"{float(preferred_row.get('recall_mean', np.nan)):.3f}",
            "description": "Mean cross-validation recall for the preferred model.",
        },
        {
            "kpi_group": "model_performance",
            "metric": "cv_f1_mean",
            "value": float(preferred_row.get("f1_mean", np.nan)),
            "display_value": f"{float(preferred_row.get('f1_mean', np.nan)):.3f}",
            "description": "Mean cross-validation F1-score for the preferred model.",
        },
        {
            "kpi_group": "model_performance",
            "metric": "cv_roc_auc_mean",
            "value": float(preferred_row.get("roc_auc_mean", np.nan)),
            "display_value": f"{float(preferred_row.get('roc_auc_mean', np.nan)):.3f}",
            "description": "Mean cross-validation ROC-AUC for the preferred model.",
        },
        {
            "kpi_group": "prediction_output",
            "metric": "rural_zone_count",
            "value": int(len(predictions)),
            "display_value": str(int(len(predictions))),
            "description": "Number of rural Data Zones scored by the Version 1 model.",
        },
        {
            "kpi_group": "prediction_output",
            "metric": "actual_underserved_count",
            "value": actual_positive,
            "display_value": str(actual_positive),
            "description": "Number of rural zones labelled as underserved in the baseline rule-based label.",
        },
        {
            "kpi_group": "prediction_output",
            "metric": "critical_underserved_count",
            "value": int(predictions[critical_col].sum()) if critical_col else np.nan,
            "display_value": str(int(predictions[critical_col].sum())) if critical_col else "N/A",
            "description": "Number of rural zones labelled as critically underserved.",
        },
        {
            "kpi_group": "prediction_output",
            "metric": "predicted_underserved_count",
            "value": predicted_positive,
            "display_value": str(predicted_positive),
            "description": "Number of rural zones predicted as underserved by the preferred model.",
        },
        {
            "kpi_group": "prediction_output",
            "metric": "high_risk_band_count",
            "value": high_risk_count,
            "display_value": str(high_risk_count),
            "description": "Number of rural zones in the high-risk probability band.",
        },
        {
            "kpi_group": "prediction_output",
            "metric": "mean_prediction_probability",
            "value": float(predictions[prob_col].mean()),
            "display_value": f"{float(predictions[prob_col].mean()):.3f}",
            "description": "Mean preferred-model probability across rural Data Zones.",
        },
    ]

    return pd.DataFrame(rows)


def build_model_comparison(cv_summary: pd.DataFrame, model_summary: dict) -> pd.DataFrame:
    preferred_model = model_summary.get("preferred_model", {}).get("model_name", "unknown")

    out = cv_summary.copy()
    out["preferred_model_flag"] = (out["model_name"] == preferred_model).astype(int)

    preferred_order = [
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
    preferred_order = [col for col in preferred_order if col in out.columns]

    return out[preferred_order].copy()


def build_feature_importance(
    rf_importance: pd.DataFrame,
    logistic_effects: pd.DataFrame,
    preferred_model_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    combined_rows = []

    if not rf_importance.empty:
        rf = rf_importance.copy()
        rf["model_name"] = "random_forest"
        rf["importance_metric"] = "feature_importance"
        rf["score"] = rf["importance"]
        rf = rf.sort_values("score", ascending=False).reset_index(drop=True)
        rf["rank"] = rf.index + 1
        combined_rows.append(rf[["model_name", "rank", "feature", "importance_metric", "score"]])

    if not logistic_effects.empty:
        logit = logistic_effects.copy()
        logit["model_name"] = "scaled_logistic_regression"
        logit["importance_metric"] = "absolute_coefficient"
        logit["score"] = logit["abs_coefficient"]
        logit = logit.sort_values("score", ascending=False).reset_index(drop=True)
        logit["rank"] = logit.index + 1
        combined_rows.append(logit[["model_name", "rank", "feature", "importance_metric", "score"]])

    if combined_rows:
        combined = pd.concat(combined_rows, ignore_index=True)
    else:
        combined = pd.DataFrame(columns=["model_name", "rank", "feature", "importance_metric", "score"])

    preferred = combined[combined["model_name"] == preferred_model_name].copy()
    if preferred.empty:
        preferred = combined.copy()

    preferred_top10 = (
        preferred.sort_values(["model_name", "rank"])
        .groupby("model_name")
        .head(10)
        .reset_index(drop=True)
    )

    combined_top10 = (
        combined.sort_values(["model_name", "rank"])
        .groupby("model_name")
        .head(10)
        .reset_index(drop=True)
    )

    return preferred_top10, combined_top10


def resolve_confusion_model_name(confusion_df: pd.DataFrame, preferred_model_name: str) -> str:
    available_models = set(confusion_df["model_name"].astype(str).unique())

    if preferred_model_name in available_models:
        return preferred_model_name

    alias_map = {
        "scaled_logistic_regression": "logistic_regression",
        "logistic_regression": "scaled_logistic_regression",
    }

    alias = alias_map.get(preferred_model_name)
    if alias in available_models:
        return alias

    if "random_forest" in available_models:
        return "random_forest"

    return sorted(available_models)[0]


def build_confusion_matrix_from_saved(
    saved_confusion: pd.DataFrame,
    preferred_model_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Uses the saved held-out/test confusion matrix from baseline_model_results.
    This avoids the misleading perfect-looking matrix created from full-data predictions.
    """
    required_cols = ["model_name", "pred_0", "pred_1"]
    missing = [col for col in required_cols if col not in saved_confusion.columns]
    if missing:
        raise ValueError(f"Missing required columns in saved confusion matrix: {missing}")

    resolved_model_name = resolve_confusion_model_name(saved_confusion, preferred_model_name)

    selected = saved_confusion[
        saved_confusion["model_name"].astype(str) == resolved_model_name
    ].copy()

    selected = selected.reset_index(drop=True)

    if "actual_class" not in selected.columns:
        # Existing baseline confusion file has two rows per model:
        # row 0 = actual class 0, row 1 = actual class 1.
        selected["actual_class"] = selected.groupby("model_name").cumcount()

    selected["actual_class"] = selected["actual_class"].astype(int)
    selected["pred_0"] = pd.to_numeric(selected["pred_0"], errors="coerce").fillna(0).astype(int)
    selected["pred_1"] = pd.to_numeric(selected["pred_1"], errors="coerce").fillna(0).astype(int)

    rows = []

    for _, row in selected.iterrows():
        actual_class = int(row["actual_class"])
        actual_total = int(row["pred_0"] + row["pred_1"])

        for predicted_class, pred_col in [(0, "pred_0"), (1, "pred_1")]:
            count = int(row[pred_col])

            if actual_class == 0 and predicted_class == 0:
                cell_type = "True Negative"
            elif actual_class == 0 and predicted_class == 1:
                cell_type = "False Positive"
            elif actual_class == 1 and predicted_class == 0:
                cell_type = "False Negative"
            else:
                cell_type = "True Positive"

            rows.append(
                {
                    "confusion_source": "held_out_test_split",
                    "display_model_name": preferred_model_name,
                    "source_model_name": resolved_model_name,
                    "actual_class": actual_class,
                    "predicted_class": predicted_class,
                    "cell_type": cell_type,
                    "count": count,
                    "actual_class_total": actual_total,
                    "share_of_actual_class": count / actual_total if actual_total > 0 else np.nan,
                }
            )

    long_df = pd.DataFrame(rows)

    wide_df = (
        selected[["model_name", "actual_class", "pred_0", "pred_1"]]
        .rename(
            columns={
                "model_name": "source_model_name",
                "pred_0": "predicted_0",
                "pred_1": "predicted_1",
            }
        )
        .copy()
    )

    wide_df["display_model_name"] = preferred_model_name
    wide_df["confusion_source"] = "held_out_test_split"

    wide_df = wide_df[
        [
            "confusion_source",
            "display_model_name",
            "source_model_name",
            "actual_class",
            "predicted_0",
            "predicted_1",
        ]
    ]

    return long_df, wide_df


def build_risk_band_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    prob_col = first_existing_col(
        predictions,
        ["preferred_model_probability", "preferred_probability", "random_forest_probability"],
    )
    pred_col = first_existing_col(
        predictions,
        ["preferred_model_predicted_class", "preferred_predicted_class"],
    )
    risk_band_col = first_existing_col(
        predictions,
        ["preferred_risk_band", "risk_band"],
    )

    grouped = (
        predictions.groupby(risk_band_col, dropna=False)
        .agg(
            zone_count=("dz_code_2022", "count"),
            mean_probability=(prob_col, "mean"),
            min_probability=(prob_col, "min"),
            max_probability=(prob_col, "max"),
            predicted_underserved_count=(pred_col, "sum"),
            actual_underserved_count=("underserved_baseline", "sum"),
            critical_underserved_count=("critical_underserved_baseline", "sum"),
        )
        .reset_index()
        .rename(columns={risk_band_col: "risk_band"})
    )

    grouped["zone_share"] = grouped["zone_count"] / grouped["zone_count"].sum()
    grouped["actual_underserved_share"] = grouped["actual_underserved_count"] / grouped["zone_count"]
    grouped["critical_underserved_share"] = grouped["critical_underserved_count"] / grouped["zone_count"]

    risk_order = {"low": 1, "medium": 2, "high": 3}
    grouped["risk_order"] = grouped["risk_band"].astype(str).str.lower().map(risk_order).fillna(99)
    grouped = grouped.sort_values("risk_order").drop(columns=["risk_order"]).reset_index(drop=True)

    return grouped


def build_ur6_prediction_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    prob_col = first_existing_col(
        predictions,
        ["preferred_model_probability", "preferred_probability", "random_forest_probability"],
    )
    pred_col = first_existing_col(
        predictions,
        ["preferred_model_predicted_class", "preferred_predicted_class"],
    )
    risk_band_col = first_existing_col(
        predictions,
        ["preferred_risk_band", "risk_band"],
        required=False,
    )

    mean_bank_agg = (
        ("dist_to_nearest_bank_km", "mean")
        if "dist_to_nearest_bank_km" in predictions.columns
        else (prob_col, "mean")
    )
    mean_any_agg = (
        ("dist_to_nearest_any_access_point_km", "mean")
        if "dist_to_nearest_any_access_point_km" in predictions.columns
        else (prob_col, "mean")
    )

    grouped = (
        predictions.groupby(["ur6_name", "ur8_name"], dropna=False)
        .agg(
            rural_zone_count=("dz_code_2022", "count"),
            mean_probability=(prob_col, "mean"),
            median_probability=(prob_col, "median"),
            predicted_underserved_count=(pred_col, "sum"),
            baseline_underserved_count=("underserved_baseline", "sum"),
            critical_underserved_count=("critical_underserved_baseline", "sum"),
            mean_bank_distance_km=mean_bank_agg,
            mean_any_access_distance_km=mean_any_agg,
        )
        .reset_index()
    )

    grouped["predicted_underserved_share"] = grouped["predicted_underserved_count"] / grouped["rural_zone_count"]
    grouped["baseline_underserved_share"] = grouped["baseline_underserved_count"] / grouped["rural_zone_count"]
    grouped["critical_underserved_share"] = grouped["critical_underserved_count"] / grouped["rural_zone_count"]

    if risk_band_col:
        high_risk = (
            predictions[predictions[risk_band_col].astype(str).str.lower() == "high"]
            .groupby(["ur6_name", "ur8_name"], dropna=False)
            .size()
            .reset_index(name="high_risk_band_count")
        )
        grouped = grouped.merge(high_risk, on=["ur6_name", "ur8_name"], how="left")
        grouped["high_risk_band_count"] = grouped["high_risk_band_count"].fillna(0).astype(int)
        grouped["high_risk_band_share"] = grouped["high_risk_band_count"] / grouped["rural_zone_count"]

    return grouped.sort_values("mean_probability", ascending=False).reset_index(drop=True)


def build_map_ready_predictions(
    predictions: pd.DataFrame,
    zones_coords: pd.DataFrame,
) -> pd.DataFrame:
    prob_col = first_existing_col(
        predictions,
        ["preferred_model_probability", "preferred_probability", "random_forest_probability"],
    )
    pred_col = first_existing_col(
        predictions,
        ["preferred_model_predicted_class", "preferred_predicted_class"],
    )
    risk_band_col = first_existing_col(
        predictions,
        ["preferred_risk_band", "risk_band"],
        required=False,
    )

    out = predictions.copy()
    out = out.merge(
        zones_coords[["dz_code_2022", "longitude", "latitude"]],
        on="dz_code_2022",
        how="left",
        validate="1:1",
    )

    out = out.sort_values(prob_col, ascending=False).reset_index(drop=True)
    out["risk_rank"] = out.index + 1

    keep_cols = [
        "risk_rank",
        "dz_code_2022",
        "dz_name_2022",
        "ur6_name",
        "ur8_name",
        "is_rural",
        "is_accessible_rural",
        "is_remote_rural",
        prob_col,
        pred_col,
        risk_band_col,
        "underserved_baseline",
        "critical_underserved_baseline",
        "dist_to_nearest_bank_km",
        "dist_to_nearest_atm_km",
        "dist_to_nearest_post_office_km",
        "dist_to_nearest_any_access_point_km",
        "population_total",
        "older_population_65_plus",
        "older_population_share",
        "active_simd_rank_overall",
        "active_access_domain_rank",
        "longitude",
        "latitude",
    ]

    keep_cols = [col for col in keep_cols if col is not None and col in out.columns]
    return out[keep_cols].copy()


def build_top_100_high_risk(map_ready: pd.DataFrame) -> pd.DataFrame:
    prob_col = first_existing_col(
        map_ready,
        ["preferred_model_probability", "preferred_probability", "random_forest_probability"],
    )

    return (
        map_ready.sort_values(prob_col, ascending=False)
        .head(100)
        .reset_index(drop=True)
    )


def main() -> None:
    print("Starting enhanced Dashboard 3 export build...")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    predictions_path = PROCESSED_DIR / "ml" / "prediction_outputs" / "rural_zone_prediction_outputs.csv"
    cv_summary_path = PROCESSED_DIR / "ml" / "refined_model_results" / "refined_model_cv_summary.csv"
    model_summary_path = PROCESSED_DIR / "ml" / "refined_model_results" / "refined_model_summary.json"
    rf_importance_path = PROCESSED_DIR / "ml" / "refined_model_results" / "random_forest_feature_importance.csv"
    logistic_effects_path = PROCESSED_DIR / "ml" / "refined_model_results" / "scaled_logistic_feature_effects.csv"
    saved_confusion_path = PROCESSED_DIR / "ml" / "baseline_model_results" / "baseline_model_confusion_matrices.csv"
    zones_path = PROCESSED_DIR / "geography" / "zones_master_2022.gpkg"

    print(f"Loading V1 rural prediction outputs from: {predictions_path}")
    predictions = load_csv(predictions_path)

    print(f"Loading refined model CV summary from: {cv_summary_path}")
    cv_summary = load_csv(cv_summary_path)

    print(f"Loading refined model summary from: {model_summary_path}")
    model_summary = load_json(model_summary_path)

    print(f"Loading random forest feature importance from: {rf_importance_path}")
    rf_importance = load_csv(rf_importance_path)

    print(f"Loading scaled logistic feature effects from: {logistic_effects_path}")
    logistic_effects = load_csv(logistic_effects_path)

    print(f"Loading saved held-out/test confusion matrix from: {saved_confusion_path}")
    saved_confusion = load_csv(saved_confusion_path)

    print(f"Loading zone geometry coordinates from: {zones_path}")
    zones_coords = read_zones_with_coordinates(zones_path)

    print(f"Prediction rows loaded: {len(predictions)}")
    print(f"CV summary rows loaded: {len(cv_summary)}")
    print(f"Random forest feature rows loaded: {len(rf_importance)}")
    print(f"Logistic feature rows loaded: {len(logistic_effects)}")
    print(f"Saved confusion matrix rows loaded: {len(saved_confusion)}")
    print(f"Zone coordinate rows loaded: {len(zones_coords)}")

    preferred_model_name = model_summary.get("preferred_model", {}).get("model_name", "unknown")

    model_kpis = build_model_kpis(predictions, cv_summary, model_summary)
    model_comparison = build_model_comparison(cv_summary, model_summary)

    preferred_feature_top10, combined_feature_top10 = build_feature_importance(
        rf_importance=rf_importance,
        logistic_effects=logistic_effects,
        preferred_model_name=preferred_model_name,
    )

    confusion_long, confusion_wide = build_confusion_matrix_from_saved(
        saved_confusion=saved_confusion,
        preferred_model_name=preferred_model_name,
    )

    risk_band_summary = build_risk_band_summary(predictions)
    ur6_prediction_summary = build_ur6_prediction_summary(predictions)
    map_ready = build_map_ready_predictions(predictions, zones_coords)
    top_100_high_risk = build_top_100_high_risk(map_ready)

    model_kpis_path = OUTPUT_DIR / "dashboard3_model_kpis.csv"
    model_comparison_path = OUTPUT_DIR / "dashboard3_model_comparison.csv"
    preferred_features_path = OUTPUT_DIR / "dashboard3_preferred_feature_importance_top10.csv"
    combined_features_path = OUTPUT_DIR / "dashboard3_all_model_feature_importance_top10.csv"
    confusion_long_path = OUTPUT_DIR / "dashboard3_confusion_matrix_long.csv"
    confusion_wide_path = OUTPUT_DIR / "dashboard3_confusion_matrix_wide.csv"
    risk_band_summary_path = OUTPUT_DIR / "dashboard3_risk_band_summary.csv"
    ur6_summary_path = OUTPUT_DIR / "dashboard3_ur6_prediction_summary.csv"
    map_ready_path = OUTPUT_DIR / "dashboard3_underserved_map_ready.csv"
    top_100_path = OUTPUT_DIR / "dashboard3_top_100_high_risk_zones.csv"

    model_kpis.to_csv(model_kpis_path, index=False)
    model_comparison.to_csv(model_comparison_path, index=False)
    preferred_feature_top10.to_csv(preferred_features_path, index=False)
    combined_feature_top10.to_csv(combined_features_path, index=False)
    confusion_long.to_csv(confusion_long_path, index=False)
    confusion_wide.to_csv(confusion_wide_path, index=False)
    risk_band_summary.to_csv(risk_band_summary_path, index=False)
    ur6_prediction_summary.to_csv(ur6_summary_path, index=False)
    map_ready.to_csv(map_ready_path, index=False)
    top_100_high_risk.to_csv(top_100_path, index=False)

    print("\n--- Enhanced Dashboard 3 Export Summary ---")
    print(f"Model KPI rows: {len(model_kpis)}")
    print(f"Model comparison rows: {len(model_comparison)}")
    print(f"Preferred feature top 10 rows: {len(preferred_feature_top10)}")
    print(f"Combined feature top 10 rows: {len(combined_feature_top10)}")
    print(f"Confusion matrix long rows: {len(confusion_long)}")
    print(f"Confusion matrix wide rows: {len(confusion_wide)}")
    print(f"Risk band summary rows: {len(risk_band_summary)}")
    print(f"UR6 prediction summary rows: {len(ur6_prediction_summary)}")
    print(f"Map-ready underserved rows: {len(map_ready)}")
    print(f"Top 100 high-risk rows: {len(top_100_high_risk)}")

    print("\nDashboard 3 model KPI preview:")
    print(model_kpis)

    print("\nDashboard 3 model comparison preview:")
    print(model_comparison)

    print("\nDashboard 3 preferred feature importance preview:")
    print(preferred_feature_top10)

    print("\nDashboard 3 corrected held-out/test confusion matrix preview:")
    print(confusion_long)

    print("\nEnhanced Dashboard 3 export build completed successfully.")
    print(f"Model KPI CSV: {model_kpis_path}")
    print(f"Model comparison CSV: {model_comparison_path}")
    print(f"Preferred feature importance CSV: {preferred_features_path}")
    print(f"All-model feature importance CSV: {combined_features_path}")
    print(f"Corrected confusion matrix long CSV: {confusion_long_path}")
    print(f"Corrected confusion matrix wide CSV: {confusion_wide_path}")
    print(f"Risk band summary CSV: {risk_band_summary_path}")
    print(f"UR6 prediction summary CSV: {ur6_summary_path}")
    print(f"Map-ready underserved CSV: {map_ready_path}")
    print(f"Top 100 high-risk CSV: {top_100_path}")


if __name__ == "__main__":
    main()