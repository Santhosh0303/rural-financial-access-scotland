from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd

from config.paths import PROCESSED_DIR


OUTPUT_DIR = PROCESSED_DIR / "dashboard_exports" / "dashboard4_enhanced"


INTERVENTION_COST_UNITS = {
    "new_atm_candidate": 1.0,
    "new_post_office_access_candidate": 2.0,
    "new_bank_access_candidate": 3.0,
    "multi_service_access_candidate": 4.0,
}

INTERVENTION_LABELS = {
    "new_atm_candidate": "New ATM candidate",
    "new_post_office_access_candidate": "New post office access candidate",
    "new_bank_access_candidate": "New bank access candidate",
    "multi_service_access_candidate": "Multi-service access candidate",
}

TIER_ORDER = {
    "tier_1_critical": 1,
    "tier_2_high_priority": 2,
    "tier_3_watchlist": 3,
}


def load_csv(path: Path, required: bool = True) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Missing required CSV file: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def read_zone_coordinates(zones_path: Path) -> pd.DataFrame:
    zones = gpd.read_file(zones_path, layer="zones_master_2022")

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


def first_existing_col(df: pd.DataFrame, candidates: list[str], required: bool = False) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    if required:
        raise ValueError(f"None of these columns were found: {candidates}")
    return None


def numeric_series(df: pd.DataFrame, col: str | None, default: float = 0.0) -> pd.Series:
    if col is None or col not in df.columns:
        return pd.Series(default, index=df.index)
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def merge_intervention_and_simulation(
    interventions: pd.DataFrame,
    simulation: pd.DataFrame,
) -> pd.DataFrame:
    if "dz_code_2022" not in interventions.columns:
        raise ValueError("scenario_interventions_all.csv is missing dz_code_2022.")
    if "dz_code_2022" not in simulation.columns:
        raise ValueError("scenario_simulation_baseline_all.csv is missing dz_code_2022.")

    merged = simulation.copy()

    extra_cols = [
        col for col in interventions.columns
        if col != "dz_code_2022" and col not in merged.columns
    ]

    if extra_cols:
        merged = merged.merge(
            interventions[["dz_code_2022"] + extra_cols],
            on="dz_code_2022",
            how="left",
            validate="1:1",
        )

    # If simulation lacks some core descriptive columns, fill from interventions.
    for col in [
        "dz_name_2022",
        "ur6_name",
        "ur8_name",
        "recommended_intervention",
        "primary_gap_service",
        "intervention_tier",
        "candidate_reason_fixed",
        "scenario_priority_rank",
    ]:
        if col not in merged.columns and col in interventions.columns:
            merged = merged.merge(
                interventions[["dz_code_2022", col]],
                on="dz_code_2022",
                how="left",
                validate="1:1",
            )

    return merged


def add_policy_fields(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "recommended_intervention" not in out.columns:
        out["recommended_intervention"] = "unknown_intervention"

    if "intervention_tier" not in out.columns:
        out["intervention_tier"] = "unknown_tier"

    out["intervention_display_name"] = (
        out["recommended_intervention"]
        .astype(str)
        .map(INTERVENTION_LABELS)
        .fillna(out["recommended_intervention"].astype(str))
    )

    out["relative_cost_units"] = (
        out["recommended_intervention"]
        .astype(str)
        .map(INTERVENTION_COST_UNITS)
        .fillna(2.0)
    )

    out["intervention_tier_order"] = (
        out["intervention_tier"]
        .astype(str)
        .map(TIER_ORDER)
        .fillna(99)
        .astype(int)
    )

    any_improvement_col = first_existing_col(
        out,
        [
            "any_access_point_km_improvement",
            "any_km_improvement",
            "access_point_km_improvement",
        ],
    )
    bank_improvement_col = first_existing_col(out, ["bank_km_improvement"])
    atm_improvement_col = first_existing_col(out, ["atm_km_improvement"])
    post_improvement_col = first_existing_col(out, ["post_office_km_improvement"])

    out["any_access_point_km_improvement_clean"] = numeric_series(out, any_improvement_col)
    out["bank_km_improvement_clean"] = numeric_series(out, bank_improvement_col)
    out["atm_km_improvement_clean"] = numeric_series(out, atm_improvement_col)
    out["post_office_km_improvement_clean"] = numeric_series(out, post_improvement_col)

    critical_col = first_existing_col(out, ["critical_underserved_baseline"], required=False)
    underserved_col = first_existing_col(out, ["underserved_baseline"], required=False)

    out["critical_priority_weight"] = np.where(numeric_series(out, critical_col) > 0, 1.5, 1.0)
    out["underserved_priority_weight"] = np.where(numeric_series(out, underserved_col) > 0, 1.25, 1.0)

    out["priority_benefit_score"] = (
        out["any_access_point_km_improvement_clean"]
        * out["critical_priority_weight"]
        * out["underserved_priority_weight"]
    )

    out["access_gain_per_relative_cost_unit"] = np.where(
        out["relative_cost_units"] > 0,
        out["priority_benefit_score"] / out["relative_cost_units"],
        np.nan,
    )

    sort_cols = [
        "intervention_tier_order",
        "priority_benefit_score",
        "any_access_point_km_improvement_clean",
        "bank_km_improvement_clean",
    ]

    out = out.sort_values(
        by=sort_cols,
        ascending=[True, False, False, False],
    ).reset_index(drop=True)

    out["policy_priority_rank_enhanced"] = out.index + 1

    return out


def add_before_after_distance_fields(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    distance_specs = [
        {
            "service": "bank",
            "baseline_candidates": [
                "dist_to_nearest_bank_km",
                "baseline_bank_km",
                "current_bank_km",
                "current_dist_to_nearest_bank_km",
            ],
            "improvement_col": "bank_km_improvement_clean",
        },
        {
            "service": "atm",
            "baseline_candidates": [
                "dist_to_nearest_atm_km",
                "baseline_atm_km",
                "current_atm_km",
                "current_dist_to_nearest_atm_km",
            ],
            "improvement_col": "atm_km_improvement_clean",
        },
        {
            "service": "post_office",
            "baseline_candidates": [
                "dist_to_nearest_post_office_km",
                "baseline_post_office_km",
                "current_post_office_km",
                "current_dist_to_nearest_post_office_km",
            ],
            "improvement_col": "post_office_km_improvement_clean",
        },
        {
            "service": "any_access_point",
            "baseline_candidates": [
                "dist_to_nearest_any_access_point_km",
                "baseline_any_access_point_km",
                "current_any_access_point_km",
                "current_dist_to_nearest_any_access_point_km",
            ],
            "improvement_col": "any_access_point_km_improvement_clean",
        },
    ]

    for spec in distance_specs:
        service = spec["service"]
        baseline_col = first_existing_col(out, spec["baseline_candidates"])
        improvement_col = spec["improvement_col"]

        if baseline_col is not None:
            out[f"current_{service}_distance_km"] = numeric_series(out, baseline_col)
            out[f"projected_{service}_distance_km"] = (
                out[f"current_{service}_distance_km"] - numeric_series(out, improvement_col)
            ).clip(lower=0)

            out[f"{service}_pct_improvement_recomputed"] = np.where(
                out[f"current_{service}_distance_km"] > 0,
                (
                    numeric_series(out, improvement_col)
                    / out[f"current_{service}_distance_km"]
                ) * 100,
                np.nan,
            )
        else:
            out[f"current_{service}_distance_km"] = np.nan
            out[f"projected_{service}_distance_km"] = np.nan
            out[f"{service}_pct_improvement_recomputed"] = np.nan

    return out


def build_kpi_table(df: pd.DataFrame) -> pd.DataFrame:
    tier1_count = int((df["intervention_tier"].astype(str) == "tier_1_critical").sum())
    tier2_count = int((df["intervention_tier"].astype(str) == "tier_2_high_priority").sum())
    tier3_count = int((df["intervention_tier"].astype(str) == "tier_3_watchlist").sum())

    total_relative_cost = float(df["relative_cost_units"].sum())
    total_priority_benefit = float(df["priority_benefit_score"].sum())

    rows = [
        {
            "kpi_group": "intervention_design",
            "metric": "total_intervention_candidates",
            "value": int(len(df)),
            "display_value": str(int(len(df))),
            "description": "Total rural zones included in the intervention design shortlist.",
        },
        {
            "kpi_group": "intervention_design",
            "metric": "tier_1_critical_candidates",
            "value": tier1_count,
            "display_value": str(tier1_count),
            "description": "Critical intervention candidates with the strongest access need.",
        },
        {
            "kpi_group": "intervention_design",
            "metric": "tier_2_high_priority_candidates",
            "value": tier2_count,
            "display_value": str(tier2_count),
            "description": "High-priority intervention candidates.",
        },
        {
            "kpi_group": "intervention_design",
            "metric": "tier_3_watchlist_candidates",
            "value": tier3_count,
            "display_value": str(tier3_count),
            "description": "Watchlist candidates for monitoring or later intervention.",
        },
        {
            "kpi_group": "access_improvement",
            "metric": "mean_any_access_improvement_km",
            "value": float(df["any_access_point_km_improvement_clean"].mean()),
            "display_value": f"{df['any_access_point_km_improvement_clean'].mean():.2f} km",
            "description": "Mean improvement in distance to the nearest financial access point under the scenario.",
        },
        {
            "kpi_group": "access_improvement",
            "metric": "max_any_access_improvement_km",
            "value": float(df["any_access_point_km_improvement_clean"].max()),
            "display_value": f"{df['any_access_point_km_improvement_clean'].max():.2f} km",
            "description": "Largest simulated improvement in distance to the nearest access point.",
        },
        {
            "kpi_group": "access_improvement",
            "metric": "mean_bank_improvement_km",
            "value": float(df["bank_km_improvement_clean"].mean()),
            "display_value": f"{df['bank_km_improvement_clean'].mean():.2f} km",
            "description": "Mean simulated bank-distance improvement.",
        },
        {
            "kpi_group": "policy_planning",
            "metric": "total_relative_cost_units",
            "value": total_relative_cost,
            "display_value": f"{total_relative_cost:.0f}",
            "description": "Illustrative relative implementation burden units, not monetary cost.",
        },
        {
            "kpi_group": "policy_planning",
            "metric": "priority_benefit_per_cost_unit",
            "value": total_priority_benefit / total_relative_cost if total_relative_cost > 0 else np.nan,
            "display_value": f"{total_priority_benefit / total_relative_cost:.2f}" if total_relative_cost > 0 else "N/A",
            "description": "Illustrative access-benefit score divided by relative implementation burden.",
        },
    ]

    return pd.DataFrame(rows)


def build_intervention_counts(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["recommended_intervention", "intervention_display_name"], dropna=False)
        .agg(
            candidate_count=("dz_code_2022", "count"),
            mean_any_access_improvement_km=("any_access_point_km_improvement_clean", "mean"),
            mean_bank_improvement_km=("bank_km_improvement_clean", "mean"),
            mean_atm_improvement_km=("atm_km_improvement_clean", "mean"),
            mean_post_office_improvement_km=("post_office_km_improvement_clean", "mean"),
            total_priority_benefit_score=("priority_benefit_score", "sum"),
            total_relative_cost_units=("relative_cost_units", "sum"),
        )
        .reset_index()
    )

    grouped["access_gain_per_relative_cost_unit"] = np.where(
        grouped["total_relative_cost_units"] > 0,
        grouped["total_priority_benefit_score"] / grouped["total_relative_cost_units"],
        np.nan,
    )

    return grouped.sort_values("candidate_count", ascending=False).reset_index(drop=True)


def build_tier_summary(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["intervention_tier", "intervention_tier_order"], dropna=False)
        .agg(
            candidate_count=("dz_code_2022", "count"),
            mean_any_access_improvement_km=("any_access_point_km_improvement_clean", "mean"),
            mean_bank_improvement_km=("bank_km_improvement_clean", "mean"),
            total_priority_benefit_score=("priority_benefit_score", "sum"),
            total_relative_cost_units=("relative_cost_units", "sum"),
        )
        .reset_index()
    )

    grouped["access_gain_per_relative_cost_unit"] = np.where(
        grouped["total_relative_cost_units"] > 0,
        grouped["total_priority_benefit_score"] / grouped["total_relative_cost_units"],
        np.nan,
    )

    return grouped.sort_values("intervention_tier_order").reset_index(drop=True)


def build_ur6_summary(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["ur6_name", "ur8_name"], dropna=False)
        .agg(
            candidate_count=("dz_code_2022", "count"),
            tier_1_count=("intervention_tier_order", lambda x: int((x == 1).sum())),
            tier_2_count=("intervention_tier_order", lambda x: int((x == 2).sum())),
            tier_3_count=("intervention_tier_order", lambda x: int((x == 3).sum())),
            mean_any_access_improvement_km=("any_access_point_km_improvement_clean", "mean"),
            mean_bank_improvement_km=("bank_km_improvement_clean", "mean"),
            mean_priority_benefit_score=("priority_benefit_score", "mean"),
            total_relative_cost_units=("relative_cost_units", "sum"),
        )
        .reset_index()
    )

    return grouped.sort_values(
        ["mean_any_access_improvement_km", "candidate_count"],
        ascending=[False, False],
    ).reset_index(drop=True)


def build_primary_gap_summary(df: pd.DataFrame) -> pd.DataFrame:
    if "primary_gap_service" not in df.columns:
        df = df.copy()
        df["primary_gap_service"] = "unknown"

    grouped = (
        df.groupby("primary_gap_service", dropna=False)
        .agg(
            candidate_count=("dz_code_2022", "count"),
            mean_any_access_improvement_km=("any_access_point_km_improvement_clean", "mean"),
            mean_bank_improvement_km=("bank_km_improvement_clean", "mean"),
            mean_atm_improvement_km=("atm_km_improvement_clean", "mean"),
            mean_post_office_improvement_km=("post_office_km_improvement_clean", "mean"),
        )
        .reset_index()
        .sort_values("candidate_count", ascending=False)
        .reset_index(drop=True)
    )

    return grouped


def build_improvement_distribution(df: pd.DataFrame) -> pd.DataFrame:
    bins = [-0.001, 1, 2, 5, 10, 15, 25, np.inf]
    labels = [
        "0-1 km",
        "1-2 km",
        "2-5 km",
        "5-10 km",
        "10-15 km",
        "15-25 km",
        "25 km+",
    ]

    out = df.copy()
    out["any_access_improvement_band"] = pd.cut(
        out["any_access_point_km_improvement_clean"],
        bins=bins,
        labels=labels,
        include_lowest=True,
    ).astype(str)

    grouped = (
        out.groupby("any_access_improvement_band", observed=False)
        .agg(
            candidate_count=("dz_code_2022", "count"),
            mean_any_access_improvement_km=("any_access_point_km_improvement_clean", "mean"),
            mean_bank_improvement_km=("bank_km_improvement_clean", "mean"),
            tier_1_count=("intervention_tier_order", lambda x: int((x == 1).sum())),
        )
        .reset_index()
    )

    grouped["candidate_share"] = grouped["candidate_count"] / grouped["candidate_count"].sum()

    return grouped


def build_policy_priority_table(df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [
        "policy_priority_rank_enhanced",
        "scenario_priority_rank",
        "dz_code_2022",
        "dz_name_2022",
        "ur6_name",
        "ur8_name",
        "recommended_intervention",
        "intervention_display_name",
        "intervention_tier",
        "primary_gap_service",
        "candidate_reason_fixed",
        "any_access_point_km_improvement_clean",
        "bank_km_improvement_clean",
        "atm_km_improvement_clean",
        "post_office_km_improvement_clean",
        "priority_benefit_score",
        "relative_cost_units",
        "access_gain_per_relative_cost_unit",
        "current_any_access_point_distance_km",
        "projected_any_access_point_distance_km",
        "any_access_point_pct_improvement_recomputed",
        "current_bank_distance_km",
        "projected_bank_distance_km",
        "bank_pct_improvement_recomputed",
        "longitude",
        "latitude",
    ]

    keep_cols = [col for col in keep_cols if col in df.columns]

    return df[keep_cols].sort_values("policy_priority_rank_enhanced").reset_index(drop=True)


def build_map_ready(df: pd.DataFrame, zone_coords: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out = out.merge(
        zone_coords[["dz_code_2022", "longitude", "latitude"]],
        on="dz_code_2022",
        how="left",
        validate="1:1",
    )

    keep_cols = [
        "policy_priority_rank_enhanced",
        "scenario_priority_rank",
        "dz_code_2022",
        "dz_name_2022",
        "ur6_name",
        "ur8_name",
        "is_rural",
        "is_accessible_rural",
        "is_remote_rural",
        "recommended_intervention",
        "intervention_display_name",
        "intervention_tier",
        "primary_gap_service",
        "candidate_reason_fixed",
        "any_access_point_km_improvement_clean",
        "bank_km_improvement_clean",
        "atm_km_improvement_clean",
        "post_office_km_improvement_clean",
        "priority_benefit_score",
        "relative_cost_units",
        "access_gain_per_relative_cost_unit",
        "current_any_access_point_distance_km",
        "projected_any_access_point_distance_km",
        "current_bank_distance_km",
        "projected_bank_distance_km",
        "longitude",
        "latitude",
    ]

    keep_cols = [col for col in keep_cols if col in out.columns]

    return out[keep_cols].copy()


def build_before_after_long(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    service_specs = [
        ("bank", "current_bank_distance_km", "projected_bank_distance_km", "bank_km_improvement_clean"),
        ("atm", "current_atm_distance_km", "projected_atm_distance_km", "atm_km_improvement_clean"),
        ("post_office", "current_post_office_distance_km", "projected_post_office_distance_km", "post_office_km_improvement_clean"),
        ("any_access_point", "current_any_access_point_distance_km", "projected_any_access_point_distance_km", "any_access_point_km_improvement_clean"),
    ]

    id_cols = [
        "policy_priority_rank_enhanced",
        "dz_code_2022",
        "dz_name_2022",
        "ur6_name",
        "ur8_name",
        "recommended_intervention",
        "intervention_tier",
    ]

    id_cols = [col for col in id_cols if col in df.columns]

    for _, row in df.iterrows():
        base_payload = {col: row[col] for col in id_cols}

        for service, current_col, projected_col, improvement_col in service_specs:
            if current_col in df.columns and projected_col in df.columns:
                current_distance = row[current_col]
                projected_distance = row[projected_col]
                improvement = row[improvement_col] if improvement_col in df.columns else np.nan

                rows.append(
                    {
                        **base_payload,
                        "service_metric": service,
                        "current_distance_km": current_distance,
                        "projected_distance_km": projected_distance,
                        "distance_improvement_km": improvement,
                        "pct_improvement": (
                            (improvement / current_distance) * 100
                            if pd.notna(current_distance) and current_distance > 0
                            else np.nan
                        ),
                    }
                )

    return pd.DataFrame(rows)


def main() -> None:
    print("Starting enhanced Dashboard 4 export build...")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    interventions_path = PROCESSED_DIR / "scenario" / "scenario_interventions_all.csv"
    simulation_path = PROCESSED_DIR / "scenario" / "scenario_simulation_baseline_all.csv"
    simulation_top100_path = PROCESSED_DIR / "scenario" / "scenario_simulation_baseline_top_100.csv"
    simulation_top20_path = PROCESSED_DIR / "scenario" / "scenario_simulation_baseline_top_20.csv"
    zones_path = PROCESSED_DIR / "geography" / "zones_master_2022.gpkg"

    print(f"Loading intervention designs from: {interventions_path}")
    interventions = load_csv(interventions_path)

    print(f"Loading scenario simulation from: {simulation_path}")
    simulation = load_csv(simulation_path)

    print(f"Loading top 100 simulation from: {simulation_top100_path}")
    simulation_top100 = load_csv(simulation_top100_path, required=False)

    print(f"Loading top 20 simulation from: {simulation_top20_path}")
    simulation_top20 = load_csv(simulation_top20_path, required=False)

    print(f"Loading zone coordinates from: {zones_path}")
    zone_coords = read_zone_coordinates(zones_path)

    print(f"Intervention rows loaded: {len(interventions)}")
    print(f"Simulation rows loaded: {len(simulation)}")
    print(f"Top 100 simulation rows loaded: {len(simulation_top100)}")
    print(f"Top 20 simulation rows loaded: {len(simulation_top20)}")
    print(f"Zone coordinate rows loaded: {len(zone_coords)}")

    merged = merge_intervention_and_simulation(interventions, simulation)
    merged = add_policy_fields(merged)
    merged = add_before_after_distance_fields(merged)

    map_ready = build_map_ready(merged, zone_coords)
    policy_priority_table = build_policy_priority_table(map_ready)

    kpi_table = build_kpi_table(merged)
    intervention_counts = build_intervention_counts(merged)
    tier_summary = build_tier_summary(merged)
    ur6_summary = build_ur6_summary(merged)
    primary_gap_summary = build_primary_gap_summary(merged)
    improvement_distribution = build_improvement_distribution(merged)
    before_after_long = build_before_after_long(merged)

    top_100_policy = policy_priority_table.head(100).copy()
    top_50_policy = policy_priority_table.head(50).copy()
    top_20_policy = policy_priority_table.head(20).copy()

    kpi_path = OUTPUT_DIR / "dashboard4_intervention_kpis.csv"
    intervention_counts_path = OUTPUT_DIR / "dashboard4_intervention_counts.csv"
    tier_summary_path = OUTPUT_DIR / "dashboard4_intervention_tier_summary.csv"
    ur6_summary_path = OUTPUT_DIR / "dashboard4_ur6_intervention_summary.csv"
    primary_gap_path = OUTPUT_DIR / "dashboard4_primary_gap_summary.csv"
    improvement_dist_path = OUTPUT_DIR / "dashboard4_improvement_distribution.csv"
    policy_priority_path = OUTPUT_DIR / "dashboard4_policy_priority_table.csv"
    top100_path = OUTPUT_DIR / "dashboard4_top_100_policy_interventions.csv"
    top50_path = OUTPUT_DIR / "dashboard4_top_50_policy_interventions.csv"
    top20_path = OUTPUT_DIR / "dashboard4_top_20_policy_interventions.csv"
    map_ready_path = OUTPUT_DIR / "dashboard4_intervention_map_ready.csv"
    before_after_path = OUTPUT_DIR / "dashboard4_before_after_accessibility_long.csv"

    kpi_table.to_csv(kpi_path, index=False)
    intervention_counts.to_csv(intervention_counts_path, index=False)
    tier_summary.to_csv(tier_summary_path, index=False)
    ur6_summary.to_csv(ur6_summary_path, index=False)
    primary_gap_summary.to_csv(primary_gap_path, index=False)
    improvement_distribution.to_csv(improvement_dist_path, index=False)
    policy_priority_table.to_csv(policy_priority_path, index=False)
    top_100_policy.to_csv(top100_path, index=False)
    top_50_policy.to_csv(top50_path, index=False)
    top_20_policy.to_csv(top20_path, index=False)
    map_ready.to_csv(map_ready_path, index=False)
    before_after_long.to_csv(before_after_path, index=False)

    print("\n--- Enhanced Dashboard 4 Export Summary ---")
    print(f"Merged intervention/simulation rows: {len(merged)}")
    print(f"KPI rows: {len(kpi_table)}")
    print(f"Intervention count rows: {len(intervention_counts)}")
    print(f"Tier summary rows: {len(tier_summary)}")
    print(f"UR6 summary rows: {len(ur6_summary)}")
    print(f"Primary gap summary rows: {len(primary_gap_summary)}")
    print(f"Improvement distribution rows: {len(improvement_distribution)}")
    print(f"Policy priority rows: {len(policy_priority_table)}")
    print(f"Top 100 policy rows: {len(top_100_policy)}")
    print(f"Top 50 policy rows: {len(top_50_policy)}")
    print(f"Top 20 policy rows: {len(top_20_policy)}")
    print(f"Map-ready intervention rows: {len(map_ready)}")
    print(f"Before/after long rows: {len(before_after_long)}")

    print("\nDashboard 4 KPI preview:")
    print(kpi_table)

    print("\nIntervention counts preview:")
    print(intervention_counts)

    print("\nTier summary preview:")
    print(tier_summary)

    print("\nTop 10 policy priority interventions:")
    preview_cols = [
        "policy_priority_rank_enhanced",
        "dz_code_2022",
        "dz_name_2022",
        "ur6_name",
        "recommended_intervention",
        "intervention_tier",
        "any_access_point_km_improvement_clean",
        "priority_benefit_score",
        "access_gain_per_relative_cost_unit",
    ]
    preview_cols = [col for col in preview_cols if col in policy_priority_table.columns]
    print(policy_priority_table[preview_cols].head(10))

    print("\nEnhanced Dashboard 4 export build completed successfully.")
    print(f"KPI CSV: {kpi_path}")
    print(f"Intervention counts CSV: {intervention_counts_path}")
    print(f"Tier summary CSV: {tier_summary_path}")
    print(f"UR6 intervention summary CSV: {ur6_summary_path}")
    print(f"Primary gap summary CSV: {primary_gap_path}")
    print(f"Improvement distribution CSV: {improvement_dist_path}")
    print(f"Policy priority table CSV: {policy_priority_path}")
    print(f"Top 100 policy interventions CSV: {top100_path}")
    print(f"Top 50 policy interventions CSV: {top50_path}")
    print(f"Top 20 policy interventions CSV: {top20_path}")
    print(f"Map-ready intervention CSV: {map_ready_path}")
    print(f"Before/after accessibility long CSV: {before_after_path}")


if __name__ == "__main__":
    main()