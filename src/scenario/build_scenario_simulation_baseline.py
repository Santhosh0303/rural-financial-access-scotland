import numpy as np
import pandas as pd
import geopandas as gpd

from config.paths import PROCESSED_DIR


def safe_pct_improvement(before: pd.Series, after: pd.Series) -> pd.Series:
    """
    Compute percentage improvement safely.
    """
    improvement = before - after
    pct = np.where(before > 0, (improvement / before) * 100, 0.0)
    return pd.Series(pct, index=before.index)


def main() -> None:
    print("Starting baseline scenario simulation build...")

    scenario_dir = PROCESSED_DIR / "scenario"
    accessibility_dir = PROCESSED_DIR / "accessibility"
    geography_path = PROCESSED_DIR / "geography" / "zones_master_2022.gpkg"

    interventions_path = scenario_dir / "scenario_interventions_all.csv"
    origins_path = accessibility_dir / "zone_origins_2022.gpkg"

    print(f"Loading intervention designs from: {interventions_path}")
    interventions = pd.read_csv(interventions_path)

    print(f"Loading zone origins from: {origins_path}")
    zone_origins = gpd.read_file(origins_path, layer="zone_origins_2022")

    print(f"Loading zone polygons from: {geography_path}")
    zones_master_2022 = gpd.read_file(geography_path, layer="zones_master_2022")

    print(f"Intervention rows loaded: {len(interventions)}")
    print(f"Zone origin rows loaded: {len(zone_origins)}")
    print(f"Zone polygon rows loaded: {len(zones_master_2022)}")

    required_cols = [
        "dz_code_2022",
        "dz_name_2022",
        "ur6_name",
        "ur8_name",
        "recommended_intervention",
        "candidate_reason_fixed",
        "intervention_tier",
        "scenario_priority_rank",
        "preferred_model_probability",
        "preferred_model_predicted_class",
        "preferred_risk_band",
        "underserved_baseline",
        "critical_underserved_baseline",
        "dist_to_nearest_bank_km",
        "dist_to_nearest_atm_km",
        "dist_to_nearest_post_office_km",
        "dist_to_nearest_any_access_point_km",
    ]

    missing = [col for col in required_cols if col not in interventions.columns]
    if missing:
        raise ValueError(f"Missing required columns in intervention table: {missing}")

    sim = interventions.copy()

    # Baseline simulation assumption:
    # A new intervention is placed at the candidate zone origin.
    # Therefore the relevant service distance for that zone becomes 0 km.
    sim["dist_to_nearest_bank_km_after"] = sim["dist_to_nearest_bank_km"]
    sim["dist_to_nearest_atm_km_after"] = sim["dist_to_nearest_atm_km"]
    sim["dist_to_nearest_post_office_km_after"] = sim["dist_to_nearest_post_office_km"]
    sim["dist_to_nearest_any_access_point_km_after"] = sim["dist_to_nearest_any_access_point_km"]

    bank_mask = sim["recommended_intervention"] == "new_bank_access_candidate"
    atm_mask = sim["recommended_intervention"] == "new_atm_candidate"
    post_mask = sim["recommended_intervention"] == "new_post_office_access_candidate"
    multi_mask = sim["recommended_intervention"] == "multi_service_access_candidate"

    sim.loc[bank_mask, "dist_to_nearest_bank_km_after"] = 0.0
    sim.loc[bank_mask, "dist_to_nearest_any_access_point_km_after"] = 0.0

    sim.loc[atm_mask, "dist_to_nearest_atm_km_after"] = 0.0
    sim.loc[atm_mask, "dist_to_nearest_any_access_point_km_after"] = 0.0

    sim.loc[post_mask, "dist_to_nearest_post_office_km_after"] = 0.0
    sim.loc[post_mask, "dist_to_nearest_any_access_point_km_after"] = 0.0

    sim.loc[multi_mask, "dist_to_nearest_bank_km_after"] = 0.0
    sim.loc[multi_mask, "dist_to_nearest_atm_km_after"] = 0.0
    sim.loc[multi_mask, "dist_to_nearest_post_office_km_after"] = 0.0
    sim.loc[multi_mask, "dist_to_nearest_any_access_point_km_after"] = 0.0

    # Improvement fields
    sim["bank_km_improvement"] = sim["dist_to_nearest_bank_km"] - sim["dist_to_nearest_bank_km_after"]
    sim["atm_km_improvement"] = sim["dist_to_nearest_atm_km"] - sim["dist_to_nearest_atm_km_after"]
    sim["post_office_km_improvement"] = (
        sim["dist_to_nearest_post_office_km"] - sim["dist_to_nearest_post_office_km_after"]
    )
    sim["any_access_point_km_improvement"] = (
        sim["dist_to_nearest_any_access_point_km"] - sim["dist_to_nearest_any_access_point_km_after"]
    )

    sim["bank_pct_improvement"] = safe_pct_improvement(
        sim["dist_to_nearest_bank_km"], sim["dist_to_nearest_bank_km_after"]
    )
    sim["atm_pct_improvement"] = safe_pct_improvement(
        sim["dist_to_nearest_atm_km"], sim["dist_to_nearest_atm_km_after"]
    )
    sim["post_office_pct_improvement"] = safe_pct_improvement(
        sim["dist_to_nearest_post_office_km"], sim["dist_to_nearest_post_office_km_after"]
    )
    sim["any_access_point_pct_improvement"] = safe_pct_improvement(
        sim["dist_to_nearest_any_access_point_km"],
        sim["dist_to_nearest_any_access_point_km_after"],
    )

    sim["total_km_improvement"] = (
        sim["bank_km_improvement"]
        + sim["atm_km_improvement"]
        + sim["post_office_km_improvement"]
        + sim["any_access_point_km_improvement"]
    )

    simulated_site_map = {
        "new_bank_access_candidate": "bank",
        "new_atm_candidate": "atm",
        "new_post_office_access_candidate": "post_office",
        "multi_service_access_candidate": "multi_service_hub",
    }
    sim["simulated_site_type"] = sim["recommended_intervention"].map(simulated_site_map).fillna("unknown")

    simulated_count_map = {
        "new_bank_access_candidate": 1,
        "new_atm_candidate": 1,
        "new_post_office_access_candidate": 1,
        "multi_service_access_candidate": 3,
    }
    sim["simulated_site_count"] = sim["recommended_intervention"].map(simulated_count_map).fillna(0).astype(int)

    # Keep sort stable and policy-readable
    sim = sim.sort_values(
        by=["scenario_priority_rank", "preferred_model_probability"],
        ascending=[True, False],
    ).reset_index(drop=True)

    top_100 = sim.head(100).copy()
    top_20 = sim.head(20).copy()

    intervention_summary = (
        sim.groupby("recommended_intervention")[
            [
                "bank_km_improvement",
                "atm_km_improvement",
                "post_office_km_improvement",
                "any_access_point_km_improvement",
            ]
        ]
        .mean()
        .reset_index()
    )

    print("\n--- Baseline Scenario Simulation Summary ---")
    print(f"Simulation rows: {len(sim)}")
    print(f"Top 100 rows: {len(top_100)}")
    print(f"Top 20 rows: {len(top_20)}")

    print("\nRecommended intervention counts:")
    print(sim["recommended_intervention"].value_counts(dropna=False))

    print("\nMean improvement by intervention type (km):")
    print(intervention_summary)

    print("\nTop 10 baseline scenario simulations:")
    print(
        sim[
            [
                "dz_code_2022",
                "dz_name_2022",
                "ur6_name",
                "ur8_name",
                "recommended_intervention",
                "simulated_site_type",
                "preferred_model_probability",
                "dist_to_nearest_any_access_point_km",
                "dist_to_nearest_any_access_point_km_after",
                "any_access_point_km_improvement",
                "any_access_point_pct_improvement",
                "intervention_tier",
            ]
        ].head(10)
    )

    # Map-ready outputs
    origin_cols = ["dz_code_2022", "origin_x", "origin_y", "geometry"]
    origin_cols = [col for col in origin_cols if col in zone_origins.columns]
    intervention_points = zone_origins[origin_cols].merge(
        sim,
        on="dz_code_2022",
        how="inner",
        validate="1:1",
    )

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
    intervention_zones = zones_master_2022[zone_cols].merge(
        sim.drop(
            columns=["dz_name_2022", "ur6_name", "ur8_name"],
            errors="ignore",
        ),
        on="dz_code_2022",
        how="inner",
        validate="1:1",
    )

    # Save outputs
    simulation_all_csv = scenario_dir / "scenario_simulation_baseline_all.csv"
    simulation_top100_csv = scenario_dir / "scenario_simulation_baseline_top_100.csv"
    simulation_top20_csv = scenario_dir / "scenario_simulation_baseline_top_20.csv"
    summary_csv = scenario_dir / "scenario_simulation_baseline_summary.csv"

    points_gpkg = scenario_dir / "scenario_simulation_baseline_points.gpkg"
    points_csv = scenario_dir / "scenario_simulation_baseline_points.csv"
    zones_gpkg = scenario_dir / "scenario_simulation_baseline_zones.gpkg"
    zones_csv = scenario_dir / "scenario_simulation_baseline_zones.csv"

    sim.to_csv(simulation_all_csv, index=False)
    top_100.to_csv(simulation_top100_csv, index=False)
    top_20.to_csv(simulation_top20_csv, index=False)
    intervention_summary.to_csv(summary_csv, index=False)

    intervention_points.to_file(
        points_gpkg,
        layer="scenario_simulation_points",
        driver="GPKG",
    )
    pd.DataFrame(intervention_points.drop(columns="geometry")).to_csv(points_csv, index=False)

    intervention_zones.to_file(
        zones_gpkg,
        layer="scenario_simulation_zones",
        driver="GPKG",
    )
    pd.DataFrame(intervention_zones.drop(columns="geometry")).to_csv(zones_csv, index=False)

    print("\nBaseline scenario simulation build completed successfully.")
    print(f"All simulation CSV: {simulation_all_csv}")
    print(f"Top 100 simulation CSV: {simulation_top100_csv}")
    print(f"Top 20 simulation CSV: {simulation_top20_csv}")
    print(f"Simulation summary CSV: {summary_csv}")
    print(f"Simulation points GPKG: {points_gpkg}")
    print(f"Simulation points CSV: {points_csv}")
    print(f"Simulation zones GPKG: {zones_gpkg}")
    print(f"Simulation zones CSV: {zones_csv}")


if __name__ == "__main__":
    main()