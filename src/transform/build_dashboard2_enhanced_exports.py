from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd

from config.paths import PROCESSED_DIR


OUTPUT_DIR = PROCESSED_DIR / "dashboard_exports" / "dashboard2_enhanced"


ACCESS_BINS_KM = [-0.001, 1, 2, 5, 10, 15, np.inf]
ACCESS_BAND_LABELS = [
    "0-1 km",
    "1-2 km",
    "2-5 km",
    "5-10 km",
    "10-15 km",
    "15 km+",
]


def read_gpkg_layer(path: Path, preferred_layer: str) -> gpd.GeoDataFrame:
    try:
        return gpd.read_file(path, layer=preferred_layer)
    except Exception:
        return gpd.read_file(path)


def ensure_km_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    distance_pairs = [
        ("dist_to_nearest_bank_m", "dist_to_nearest_bank_km"),
        ("dist_to_nearest_atm_m", "dist_to_nearest_atm_km"),
        ("dist_to_nearest_post_office_m", "dist_to_nearest_post_office_km"),
        ("dist_to_nearest_any_access_point_m", "dist_to_nearest_any_access_point_km"),
    ]

    for metres_col, km_col in distance_pairs:
        if km_col not in out.columns and metres_col in out.columns:
            out[km_col] = out[metres_col] / 1000.0

    return out


def add_access_bands(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["any_access_distance_band"] = pd.cut(
        out["dist_to_nearest_any_access_point_km"],
        bins=ACCESS_BINS_KM,
        labels=ACCESS_BAND_LABELS,
        include_lowest=True,
    ).astype(str)

    out["bank_distance_band"] = pd.cut(
        out["dist_to_nearest_bank_km"],
        bins=ACCESS_BINS_KM,
        labels=ACCESS_BAND_LABELS,
        include_lowest=True,
    ).astype(str)

    out["within_5km_any_access"] = (
        out["dist_to_nearest_any_access_point_km"] <= 5
    ).astype(int)
    out["within_10km_any_access"] = (
        out["dist_to_nearest_any_access_point_km"] <= 10
    ).astype(int)
    out["within_15km_any_access"] = (
        out["dist_to_nearest_any_access_point_km"] <= 15
    ).astype(int)

    out["within_5km_bank"] = (
        out["dist_to_nearest_bank_km"] <= 5
    ).astype(int)
    out["within_10km_bank"] = (
        out["dist_to_nearest_bank_km"] <= 10
    ).astype(int)
    out["within_15km_bank"] = (
        out["dist_to_nearest_bank_km"] <= 15
    ).astype(int)

    out["access_gap_flag_5km_any"] = (
        out["dist_to_nearest_any_access_point_km"] > 5
    ).astype(int)
    out["access_gap_flag_10km_bank"] = (
        out["dist_to_nearest_bank_km"] > 10
    ).astype(int)

    return out


def weighted_share(df: pd.DataFrame, flag_col: str, weight_col: str = "population_total") -> float:
    if weight_col not in df.columns or df[weight_col].sum() <= 0:
        return float(df[flag_col].mean())

    return float((df[flag_col] * df[weight_col]).sum() / df[weight_col].sum())


def build_kpi_table(df: pd.DataFrame) -> pd.DataFrame:
    rural_df = df[df["is_rural"] == 1].copy()

    rows = [
        {
            "kpi_group": "current_accessibility",
            "metric": "total_zones",
            "value": int(len(df)),
            "display_value": str(int(len(df))),
            "description": "Total number of 2022 Data Zones included in the accessibility analysis.",
        },
        {
            "kpi_group": "current_accessibility",
            "metric": "rural_zones",
            "value": int(len(rural_df)),
            "display_value": str(int(len(rural_df))),
            "description": "Number of rural Data Zones in the analysis.",
        },
        {
            "kpi_group": "current_accessibility",
            "metric": "mean_any_access_distance_km",
            "value": float(df["dist_to_nearest_any_access_point_km"].mean()),
            "display_value": f"{df['dist_to_nearest_any_access_point_km'].mean():.2f} km",
            "description": "Mean distance from Data Zone origin to nearest bank, ATM, or post office.",
        },
        {
            "kpi_group": "current_accessibility",
            "metric": "median_any_access_distance_km",
            "value": float(df["dist_to_nearest_any_access_point_km"].median()),
            "display_value": f"{df['dist_to_nearest_any_access_point_km'].median():.2f} km",
            "description": "Median distance from Data Zone origin to nearest bank, ATM, or post office.",
        },
        {
            "kpi_group": "coverage_threshold",
            "metric": "population_share_within_5km_any_access",
            "value": weighted_share(df, "within_5km_any_access"),
            "display_value": f"{weighted_share(df, 'within_5km_any_access') * 100:.1f}%",
            "description": "Population-weighted share of Data Zones within 5 km of any physical financial access point.",
        },
        {
            "kpi_group": "coverage_threshold",
            "metric": "population_share_within_10km_any_access",
            "value": weighted_share(df, "within_10km_any_access"),
            "display_value": f"{weighted_share(df, 'within_10km_any_access') * 100:.1f}%",
            "description": "Population-weighted share of Data Zones within 10 km of any physical financial access point.",
        },
        {
            "kpi_group": "coverage_threshold",
            "metric": "population_share_within_15km_any_access",
            "value": weighted_share(df, "within_15km_any_access"),
            "display_value": f"{weighted_share(df, 'within_15km_any_access') * 100:.1f}%",
            "description": "Population-weighted share of Data Zones within 15 km of any physical financial access point.",
        },
        {
            "kpi_group": "bank_accessibility",
            "metric": "population_share_within_10km_bank",
            "value": weighted_share(df, "within_10km_bank"),
            "display_value": f"{weighted_share(df, 'within_10km_bank') * 100:.1f}%",
            "description": "Population-weighted share of Data Zones within 10 km of a bank branch.",
        },
        {
            "kpi_group": "rural_accessibility",
            "metric": "rural_mean_any_access_distance_km",
            "value": float(rural_df["dist_to_nearest_any_access_point_km"].mean()),
            "display_value": f"{rural_df['dist_to_nearest_any_access_point_km'].mean():.2f} km",
            "description": "Mean rural Data Zone distance to nearest bank, ATM, or post office.",
        },
        {
            "kpi_group": "rural_accessibility",
            "metric": "rural_mean_bank_distance_km",
            "value": float(rural_df["dist_to_nearest_bank_km"].mean()),
            "display_value": f"{rural_df['dist_to_nearest_bank_km'].mean():.2f} km",
            "description": "Mean rural Data Zone distance to nearest bank.",
        },
    ]

    return pd.DataFrame(rows)


def build_distance_distribution(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby("any_access_distance_band", observed=False)
        .agg(
            zone_count=("dz_code_2022", "count"),
            population_total=("population_total", "sum"),
            older_population_65_plus=("older_population_65_plus", "sum"),
            mean_any_access_km=("dist_to_nearest_any_access_point_km", "mean"),
            mean_bank_km=("dist_to_nearest_bank_km", "mean"),
            rural_zone_count=("is_rural", "sum"),
        )
        .reset_index()
    )

    grouped["zone_share"] = grouped["zone_count"] / grouped["zone_count"].sum()
    grouped["population_share"] = grouped["population_total"] / grouped["population_total"].sum()

    return grouped


def build_ur6_accessibility_summary(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby("ur6_name", dropna=False)
        .agg(
            zone_count=("dz_code_2022", "count"),
            population_total=("population_total", "sum"),
            older_population_65_plus=("older_population_65_plus", "sum"),
            mean_any_access_km=("dist_to_nearest_any_access_point_km", "mean"),
            median_any_access_km=("dist_to_nearest_any_access_point_km", "median"),
            p75_any_access_km=("dist_to_nearest_any_access_point_km", lambda x: x.quantile(0.75)),
            mean_bank_km=("dist_to_nearest_bank_km", "mean"),
            median_bank_km=("dist_to_nearest_bank_km", "median"),
            p75_bank_km=("dist_to_nearest_bank_km", lambda x: x.quantile(0.75)),
            zones_within_5km_any_access=("within_5km_any_access", "sum"),
            zones_within_10km_any_access=("within_10km_any_access", "sum"),
            zones_within_15km_any_access=("within_15km_any_access", "sum"),
            zones_within_10km_bank=("within_10km_bank", "sum"),
        )
        .reset_index()
    )

    grouped["older_population_share"] = np.where(
        grouped["population_total"] > 0,
        grouped["older_population_65_plus"] / grouped["population_total"],
        np.nan,
    )

    grouped["zone_share_within_5km_any_access"] = (
        grouped["zones_within_5km_any_access"] / grouped["zone_count"]
    )
    grouped["zone_share_within_10km_any_access"] = (
        grouped["zones_within_10km_any_access"] / grouped["zone_count"]
    )
    grouped["zone_share_within_15km_any_access"] = (
        grouped["zones_within_15km_any_access"] / grouped["zone_count"]
    )
    grouped["zone_share_within_10km_bank"] = (
        grouped["zones_within_10km_bank"] / grouped["zone_count"]
    )

    return grouped.sort_values("mean_bank_km", ascending=False).reset_index(drop=True)


def build_vulnerable_segment_summary(df: pd.DataFrame) -> pd.DataFrame:
    rural_df = df[df["is_rural"] == 1].copy()

    grouped = (
        rural_df.groupby(["ur6_name", "ur8_name", "any_access_distance_band"], dropna=False, observed=False)
        .agg(
            zone_count=("dz_code_2022", "count"),
            population_total=("population_total", "sum"),
            older_population_65_plus=("older_population_65_plus", "sum"),
            mean_older_population_share=("older_population_share", "mean"),
            mean_any_access_km=("dist_to_nearest_any_access_point_km", "mean"),
            mean_bank_km=("dist_to_nearest_bank_km", "mean"),
            access_gap_5km_zone_count=("access_gap_flag_5km_any", "sum"),
            bank_gap_10km_zone_count=("access_gap_flag_10km_bank", "sum"),
        )
        .reset_index()
    )

    grouped["older_population_exposure_score"] = (
        grouped["older_population_65_plus"] * grouped["mean_any_access_km"]
    )

    grouped = grouped.sort_values(
        ["older_population_exposure_score", "mean_bank_km"],
        ascending=[False, False],
    ).reset_index(drop=True)

    return grouped


def build_top_vulnerable_zones(df: pd.DataFrame) -> pd.DataFrame:
    rural_df = df[df["is_rural"] == 1].copy()

    rural_df["older_population_access_exposure_score"] = (
        rural_df["older_population_65_plus"] *
        rural_df["dist_to_nearest_any_access_point_km"]
    )

    keep_cols = [
        "dz_code_2022",
        "dz_name_2022",
        "ur6_name",
        "ur8_name",
        "population_total",
        "older_population_65_plus",
        "older_population_share",
        "dist_to_nearest_any_access_point_km",
        "dist_to_nearest_bank_km",
        "any_access_distance_band",
        "bank_distance_band",
        "older_population_access_exposure_score",
        "latitude",
        "longitude",
    ]

    keep_cols = [col for col in keep_cols if col in rural_df.columns]

    return (
        rural_df[keep_cols]
        .sort_values(
            ["older_population_access_exposure_score", "dist_to_nearest_any_access_point_km"],
            ascending=[False, False],
        )
        .head(100)
        .reset_index(drop=True)
    )


def build_map_ready_accessibility(df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [
        "dz_code_2022",
        "dz_name_2022",
        "ur6_name",
        "ur8_name",
        "is_rural",
        "is_accessible_rural",
        "is_remote_rural",
        "population_total",
        "older_population_65_plus",
        "older_population_share",
        "dist_to_nearest_bank_km",
        "dist_to_nearest_atm_km",
        "dist_to_nearest_post_office_km",
        "dist_to_nearest_any_access_point_km",
        "any_access_distance_band",
        "bank_distance_band",
        "within_5km_any_access",
        "within_10km_any_access",
        "within_15km_any_access",
        "within_10km_bank",
        "latitude",
        "longitude",
    ]

    keep_cols = [col for col in keep_cols if col in df.columns]
    return df[keep_cols].copy()


def build_temporal_change_summary(temporal_df: pd.DataFrame) -> pd.DataFrame:
    out = temporal_df.copy()

    if "mean_bank_change_km" not in out.columns:
        out["mean_bank_change_km"] = out["mean_bank_km_2023"] - out["mean_bank_km_2019"]

    out["deterioration_share"] = np.where(
        out["zone_count"] > 0,
        out["deteriorating_zone_count"] / out["zone_count"],
        np.nan,
    )

    out["major_deterioration_share"] = np.where(
        out["zone_count"] > 0,
        out["major_deterioration_zone_count"] / out["zone_count"],
        np.nan,
    )

    if "severe_deterioration_zone_count" in out.columns:
        out["severe_deterioration_share"] = np.where(
            out["zone_count"] > 0,
            out["severe_deterioration_zone_count"] / out["zone_count"],
            np.nan,
        )

    return out


def main() -> None:
    print("Starting enhanced Dashboard 2 export build...")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    accessibility_path = PROCESSED_DIR / "accessibility" / "zone_accessibility_baseline_2022.csv"
    origins_path = PROCESSED_DIR / "accessibility" / "zone_origins_2022.gpkg"
    context_path = PROCESSED_DIR / "context" / "zone_year_context_2022.csv"
    temporal_summary_path = PROCESSED_DIR / "accessibility" / "bank_accessibility_temporal_summary_by_ur6.csv"
    temporal_top100_path = PROCESSED_DIR / "accessibility" / "bank_accessibility_top_100_worsening_rural_zones.csv"

    print(f"Loading accessibility baseline from: {accessibility_path}")
    access = pd.read_csv(accessibility_path)
    access = ensure_km_columns(access)

    print(f"Loading zone origins from: {origins_path}")
    origins = read_gpkg_layer(origins_path, "zone_origins_2022")

    print(f"Loading context from: {context_path}")
    context = pd.read_csv(context_path)

    print(f"Loading temporal summary from: {temporal_summary_path}")
    temporal_summary = pd.read_csv(temporal_summary_path)

    print(f"Loading temporal top 100 worsening rural zones from: {temporal_top100_path}")
    temporal_top100 = pd.read_csv(temporal_top100_path)

    print(f"Accessibility rows: {len(access)}")
    print(f"Origin rows: {len(origins)}")
    print(f"Context rows: {len(context)}")
    print(f"Temporal summary rows: {len(temporal_summary)}")
    print(f"Temporal top 100 rows: {len(temporal_top100)}")

    # Latest context year, expected 2023 from the harmonised context.
    latest_year = int(context["year"].max())
    context_latest = context[context["year"] == latest_year].copy()

    context_keep_cols = [
        "dz_code_2022",
        "population_total",
        "older_population_65_plus",
        "older_population_share",
        "active_simd_rank_overall",
        "active_access_domain_rank",
        "active_health_domain_rank",
        "active_income_domain_rank",
    ]
    context_keep_cols = [col for col in context_keep_cols if col in context_latest.columns]

    context_latest = (
        context_latest[context_keep_cols]
        .drop_duplicates(subset=["dz_code_2022"])
        .copy()
    )

    # Convert origins to WGS84 for Power BI maps.
    origins_wgs84 = origins.to_crs("EPSG:4326")
    origins_wgs84["longitude"] = origins_wgs84.geometry.x
    origins_wgs84["latitude"] = origins_wgs84.geometry.y

    origin_keep_cols = [
        "dz_code_2022",
        "longitude",
        "latitude",
    ]
    origins_wgs84 = origins_wgs84[origin_keep_cols].drop_duplicates(subset=["dz_code_2022"])

    # Merge master accessibility table.
    access_master = access.merge(
        context_latest,
        on="dz_code_2022",
        how="left",
        validate="1:1",
    )

    access_master = access_master.merge(
        origins_wgs84,
        on="dz_code_2022",
        how="left",
        validate="1:1",
    )

    # Fill context values defensively.
    for col in ["population_total", "older_population_65_plus", "older_population_share"]:
        if col in access_master.columns:
            access_master[col] = pd.to_numeric(access_master[col], errors="coerce").fillna(0)

    access_master = add_access_bands(access_master)

    kpi_table = build_kpi_table(access_master)
    distance_distribution = build_distance_distribution(access_master)
    ur6_summary = build_ur6_accessibility_summary(access_master)
    vulnerable_segment_summary = build_vulnerable_segment_summary(access_master)
    top_vulnerable_zones = build_top_vulnerable_zones(access_master)
    map_ready_accessibility = build_map_ready_accessibility(access_master)
    temporal_change_summary = build_temporal_change_summary(temporal_summary)

    # Clean enhanced top 100 current least accessible rural zones.
    top_least_accessible = (
        access_master[access_master["is_rural"] == 1]
        .copy()
        .sort_values(
            ["dist_to_nearest_any_access_point_km", "dist_to_nearest_bank_km"],
            ascending=[False, False],
        )
        .head(100)
        .reset_index(drop=True)
    )
    top_least_accessible["least_accessible_rank"] = top_least_accessible.index + 1

    least_cols = [
        "least_accessible_rank",
        "dz_code_2022",
        "dz_name_2022",
        "ur6_name",
        "ur8_name",
        "population_total",
        "older_population_65_plus",
        "older_population_share",
        "dist_to_nearest_any_access_point_km",
        "dist_to_nearest_bank_km",
        "dist_to_nearest_atm_km",
        "dist_to_nearest_post_office_km",
        "any_access_distance_band",
        "bank_distance_band",
        "latitude",
        "longitude",
    ]
    least_cols = [col for col in least_cols if col in top_least_accessible.columns]
    top_least_accessible = top_least_accessible[least_cols]

    # Save outputs.
    kpi_path = OUTPUT_DIR / "dashboard2_accessibility_kpis.csv"
    map_ready_path = OUTPUT_DIR / "dashboard2_accessibility_map_ready.csv"
    distance_dist_path = OUTPUT_DIR / "dashboard2_distance_distribution.csv"
    ur6_summary_path = OUTPUT_DIR / "dashboard2_accessibility_ur6_enhanced.csv"
    vulnerable_summary_path = OUTPUT_DIR / "dashboard2_vulnerable_population_segments.csv"
    top_vulnerable_path = OUTPUT_DIR / "dashboard2_top_100_vulnerable_access_zones.csv"
    top_least_path = OUTPUT_DIR / "dashboard2_top_100_least_accessible_rural_zones_enhanced.csv"
    temporal_summary_out_path = OUTPUT_DIR / "dashboard2_temporal_bank_change_by_ur6.csv"
    temporal_top100_out_path = OUTPUT_DIR / "dashboard2_top_100_worsening_rural_zones.csv"

    kpi_table.to_csv(kpi_path, index=False)
    map_ready_accessibility.to_csv(map_ready_path, index=False)
    distance_distribution.to_csv(distance_dist_path, index=False)
    ur6_summary.to_csv(ur6_summary_path, index=False)
    vulnerable_segment_summary.to_csv(vulnerable_summary_path, index=False)
    top_vulnerable_zones.to_csv(top_vulnerable_path, index=False)
    top_least_accessible.to_csv(top_least_path, index=False)
    temporal_change_summary.to_csv(temporal_summary_out_path, index=False)
    temporal_top100.to_csv(temporal_top100_out_path, index=False)

    print("\n--- Enhanced Dashboard 2 Export Summary ---")
    print(f"KPI rows: {len(kpi_table)}")
    print(f"Map-ready accessibility rows: {len(map_ready_accessibility)}")
    print(f"Distance distribution rows: {len(distance_distribution)}")
    print(f"Enhanced UR6 summary rows: {len(ur6_summary)}")
    print(f"Vulnerable segment rows: {len(vulnerable_segment_summary)}")
    print(f"Top vulnerable access zones rows: {len(top_vulnerable_zones)}")
    print(f"Top least accessible rural zones rows: {len(top_least_accessible)}")
    print(f"Temporal bank change by UR6 rows: {len(temporal_change_summary)}")
    print(f"Temporal top 100 worsening rows: {len(temporal_top100)}")

    print("\nDashboard 2 KPI preview:")
    print(kpi_table)

    print("\nDistance distribution preview:")
    print(distance_distribution)

    print("\nEnhanced UR6 summary preview:")
    print(ur6_summary.head(10))

    print("\nEnhanced Dashboard 2 export build completed successfully.")
    print(f"KPI CSV: {kpi_path}")
    print(f"Map-ready accessibility CSV: {map_ready_path}")
    print(f"Distance distribution CSV: {distance_dist_path}")
    print(f"Enhanced UR6 accessibility CSV: {ur6_summary_path}")
    print(f"Vulnerable population segments CSV: {vulnerable_summary_path}")
    print(f"Top 100 vulnerable access zones CSV: {top_vulnerable_path}")
    print(f"Top 100 least accessible rural zones CSV: {top_least_path}")
    print(f"Temporal bank change by UR6 CSV: {temporal_summary_out_path}")
    print(f"Temporal top 100 worsening rural zones CSV: {temporal_top100_out_path}")


if __name__ == "__main__":
    main()