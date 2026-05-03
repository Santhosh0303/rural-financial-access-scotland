import pandas as pd

from config.paths import PROCESSED_DIR


def build_dashboard_1_service_distribution(
    services_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Dashboard 1:
    - service counts by type
    - service counts by rural class
    - rural service coverage summary
    """
    service_counts = (
        services_df.groupby("service_type", dropna=False)
        .size()
        .reset_index(name="service_count")
        .sort_values("service_count", ascending=False)
        .reset_index(drop=True)
    )

    service_by_ur6 = (
        services_df.groupby(["ur6_name", "service_type"], dropna=False)
        .size()
        .reset_index(name="service_count")
        .sort_values(["ur6_name", "service_count"], ascending=[True, False])
        .reset_index(drop=True)
    )

    rural_zone_summary = (
        predictions_df.groupby(["ur6_name", "ur8_name"], dropna=False)
        .agg(
            rural_zone_count=("dz_code_2022", "count"),
            predicted_underserved_count=("preferred_model_predicted_class", "sum"),
            baseline_underserved_count=("underserved_baseline", "sum"),
        )
        .reset_index()
    )

    rural_zone_summary["predicted_underserved_share"] = (
        rural_zone_summary["predicted_underserved_count"] / rural_zone_summary["rural_zone_count"]
    )
    rural_zone_summary["baseline_underserved_share"] = (
        rural_zone_summary["baseline_underserved_count"] / rural_zone_summary["rural_zone_count"]
    )

    return service_counts, service_by_ur6, rural_zone_summary


def build_dashboard_2_accessibility(
    zone_access_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Dashboard 2:
    - accessibility summary by UR6
    - top least-accessible rural zones
    """
    access_by_ur6 = (
        zone_access_df.groupby("ur6_name", dropna=False)
        .agg(
            zone_count=("dz_code_2022", "count"),
            mean_bank_km=("dist_to_nearest_bank_km", "mean"),
            mean_atm_km=("dist_to_nearest_atm_km", "mean"),
            mean_post_office_km=("dist_to_nearest_post_office_km", "mean"),
            mean_any_access_point_km=("dist_to_nearest_any_access_point_km", "mean"),
            median_any_access_point_km=("dist_to_nearest_any_access_point_km", "median"),
            p75_any_access_point_km=("dist_to_nearest_any_access_point_km", lambda s: s.quantile(0.75)),
        )
        .reset_index()
        .sort_values("mean_any_access_point_km", ascending=False)
        .reset_index(drop=True)
    )

    least_accessible_rural = (
        zone_access_df[zone_access_df["is_rural"] == 1]
        .copy()
        .sort_values("dist_to_nearest_any_access_point_km", ascending=False)
        .reset_index(drop=True)
    )

    least_accessible_rural["rural_access_rank"] = least_accessible_rural.index + 1

    least_accessible_rural = least_accessible_rural[
        [
            "rural_access_rank",
            "dz_code_2022",
            "dz_name_2022",
            "ur6_name",
            "ur8_name",
            "dist_to_nearest_bank_km",
            "dist_to_nearest_atm_km",
            "dist_to_nearest_post_office_km",
            "dist_to_nearest_any_access_point_km",
        ]
    ].head(100)

    return access_by_ur6, least_accessible_rural


def build_dashboard_3_underserved_predictions(
    predictions_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Dashboard 3:
    - prediction/risk summary by UR6
    - top high-risk zones
    - model class / risk band summary
    """
    prediction_by_ur6 = (
        predictions_df.groupby("ur6_name", dropna=False)
        .agg(
            rural_zone_count=("dz_code_2022", "count"),
            avg_predicted_probability=("preferred_model_probability", "mean"),
            predicted_positive_count=("preferred_model_predicted_class", "sum"),
            baseline_underserved_count=("underserved_baseline", "sum"),
            critical_underserved_count=("critical_underserved_baseline", "sum"),
        )
        .reset_index()
    )

    prediction_by_ur6["predicted_positive_share"] = (
        prediction_by_ur6["predicted_positive_count"] / prediction_by_ur6["rural_zone_count"]
    )
    prediction_by_ur6["baseline_underserved_share"] = (
        prediction_by_ur6["baseline_underserved_count"] / prediction_by_ur6["rural_zone_count"]
    )

    top_high_risk = (
        predictions_df.copy()
        .sort_values("preferred_model_probability", ascending=False)
        .reset_index(drop=True)
    )
    top_high_risk["high_risk_rank"] = top_high_risk.index + 1

    top_high_risk = top_high_risk[
        [
            "high_risk_rank",
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
    ].head(100)

    class_band_summary = pd.DataFrame(
        {
            "metric": [
                "predicted_positive_count",
                "predicted_negative_count",
                "high_risk_band_count",
                "medium_risk_band_count",
                "low_risk_band_count",
            ],
            "value": [
                int((predictions_df["preferred_model_predicted_class"] == 1).sum()),
                int((predictions_df["preferred_model_predicted_class"] == 0).sum()),
                int((predictions_df["preferred_risk_band"] == "high").sum()),
                int((predictions_df["preferred_risk_band"] == "medium").sum()),
                int((predictions_df["preferred_risk_band"] == "low").sum()),
            ],
        }
    )

    return prediction_by_ur6, top_high_risk, class_band_summary


def build_dashboard_4_scenarios(
    interventions_df: pd.DataFrame,
    simulation_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Dashboard 4:
    - intervention counts
    - mean improvement summary
    - top intervention shortlist
    """
    intervention_counts = (
        interventions_df.groupby(
            ["recommended_intervention", "intervention_tier"],
            dropna=False,
        )
        .size()
        .reset_index(name="zone_count")
        .sort_values(["intervention_tier", "zone_count"], ascending=[True, False])
        .reset_index(drop=True)
    )

    simulation_summary = (
        simulation_df.groupby("recommended_intervention", dropna=False)
        .agg(
            zone_count=("dz_code_2022", "count"),
            mean_bank_km_improvement=("bank_km_improvement", "mean"),
            mean_atm_km_improvement=("atm_km_improvement", "mean"),
            mean_post_office_km_improvement=("post_office_km_improvement", "mean"),
            mean_any_access_point_km_improvement=("any_access_point_km_improvement", "mean"),
            mean_any_access_point_pct_improvement=("any_access_point_pct_improvement", "mean"),
        )
        .reset_index()
        .sort_values("mean_any_access_point_km_improvement", ascending=False)
        .reset_index(drop=True)
    )

    top_interventions = (
        simulation_df.copy()
        .sort_values(
            ["scenario_priority_rank", "preferred_model_probability"],
            ascending=[True, False],
        )
        .reset_index(drop=True)
    )

    top_interventions = top_interventions[
        [
            "scenario_priority_rank",
            "dz_code_2022",
            "dz_name_2022",
            "ur6_name",
            "ur8_name",
            "recommended_intervention",
            "intervention_tier",
            "preferred_model_probability",
            "dist_to_nearest_any_access_point_km",
            "dist_to_nearest_any_access_point_km_after",
            "any_access_point_km_improvement",
            "any_access_point_pct_improvement",
        ]
    ].head(100)

    return intervention_counts, simulation_summary, top_interventions


def main() -> None:
    print("Starting dashboard export build...")

    processed_dir = PROCESSED_DIR
    dashboard_dir = processed_dir / "dashboard_exports"
    dashboard_dir.mkdir(parents=True, exist_ok=True)

    services_path = processed_dir / "services_current" / "service_points_current.csv"
    zone_access_path = processed_dir / "accessibility" / "zone_accessibility_baseline_2022.csv"
    predictions_path = processed_dir / "ml" / "prediction_outputs" / "rural_zone_prediction_outputs.csv"
    interventions_path = processed_dir / "scenario" / "scenario_interventions_all.csv"
    simulation_path = processed_dir / "scenario" / "scenario_simulation_baseline_all.csv"

    print(f"Loading services from: {services_path}")
    services_df = pd.read_csv(services_path)

    print(f"Loading accessibility from: {zone_access_path}")
    zone_access_df = pd.read_csv(zone_access_path)

    print(f"Loading predictions from: {predictions_path}")
    predictions_df = pd.read_csv(predictions_path)

    print(f"Loading interventions from: {interventions_path}")
    interventions_df = pd.read_csv(interventions_path)

    print(f"Loading simulation from: {simulation_path}")
    simulation_df = pd.read_csv(simulation_path)

    print(f"Services rows loaded: {len(services_df)}")
    print(f"Accessibility rows loaded: {len(zone_access_df)}")
    print(f"Prediction rows loaded: {len(predictions_df)}")
    print(f"Intervention rows loaded: {len(interventions_df)}")
    print(f"Simulation rows loaded: {len(simulation_df)}")

    d1_service_counts, d1_service_by_ur6, d1_rural_zone_summary = build_dashboard_1_service_distribution(
        services_df, predictions_df
    )
    d2_access_by_ur6, d2_least_accessible = build_dashboard_2_accessibility(zone_access_df)
    d3_prediction_by_ur6, d3_top_high_risk, d3_class_band_summary = build_dashboard_3_underserved_predictions(
        predictions_df
    )
    d4_intervention_counts, d4_sim_summary, d4_top_interventions = build_dashboard_4_scenarios(
        interventions_df, simulation_df
    )

    files_to_save = {
        "dashboard1_service_counts.csv": d1_service_counts,
        "dashboard1_service_counts_by_ur6.csv": d1_service_by_ur6,
        "dashboard1_rural_zone_summary.csv": d1_rural_zone_summary,
        "dashboard2_accessibility_by_ur6.csv": d2_access_by_ur6,
        "dashboard2_top_100_least_accessible_rural_zones.csv": d2_least_accessible,
        "dashboard3_prediction_by_ur6.csv": d3_prediction_by_ur6,
        "dashboard3_top_100_high_risk_rural_zones.csv": d3_top_high_risk,
        "dashboard3_class_band_summary.csv": d3_class_band_summary,
        "dashboard4_intervention_counts.csv": d4_intervention_counts,
        "dashboard4_simulation_summary.csv": d4_sim_summary,
        "dashboard4_top_100_interventions.csv": d4_top_interventions,
    }

    for filename, df in files_to_save.items():
        output_path = dashboard_dir / filename
        df.to_csv(output_path, index=False)

    print("\n--- Dashboard Export Summary ---")
    for filename, df in files_to_save.items():
        print(f"{filename}: {len(df)} rows")

    print("\nDashboard export build completed successfully.")
    print(f"Output folder: {dashboard_dir}")


if __name__ == "__main__":
    main()