from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd

from config.paths import PROCESSED_DIR


EXPECTED_COLUMNS = [
    "id",
    "brand_full",
    "brand_short",
    "branch_name",
    "branch_type",
    "add_one",
    "add_two",
    "suburb",
    "town",
    "region",
    "postcode",
    "long_wgs84",
    "lat_wgs84",
    "status",
    "close_month",
    "close_year",
    "open_year",
    "po_dist",
]

PRE_YEAR = 2019
POST_YEAR = 2023


def resolve_geolytix_folder(raw_dir: Path) -> Path:
    preferred_candidates = [
        raw_dir / "bank_closures Geolytix" / "GEOLYTIX - UK Open Banks",
        raw_dir / "bank_closures_Geolytix" / "GEOLYTIX - UK Open Banks",
        raw_dir / "bank_closures" / "Geolytix" / "GEOLYTIX - UK Open Banks",
        raw_dir / "bank_closures" / "GEOLYTIX - UK Open Banks",
    ]

    for candidate in preferred_candidates:
        if candidate.exists():
            return candidate

    matches = []
    for path in raw_dir.rglob("*"):
        if path.is_dir():
            path_name = path.name.lower()
            parent_name = path.parent.name.lower()
            if ("geolytix" in path_name and "open banks" in path_name) or (
                "geolytix" in parent_name and "open banks" in path_name
            ):
                matches.append(path)

    if matches:
        matches = sorted(matches)
        return matches[0]

    raise FileNotFoundError(
        "Could not find the 'GEOLYTIX - UK Open Banks' folder under data/raw."
    )


def resolve_required_file(folder: Path, filename: str) -> Path:
    direct_path = folder / filename
    if direct_path.exists():
        return direct_path

    matches = list(folder.rglob(filename))
    if matches:
        return matches[0]

    raise FileNotFoundError(f"Missing required file: {filename} under {folder}")


def clean_text_columns(df: pd.DataFrame, text_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in text_cols:
        if col in out.columns:
            out[col] = out[col].astype("string").str.strip()
    return out


def is_active_in_year(df: pd.DataFrame, snapshot_year: int) -> pd.Series:
    """
    Year-end style snapshot logic:
    - if open_year exists, branch must have opened on or before the snapshot year
    - if close_year exists, branch must close after the snapshot year to remain active
    """
    open_ok = df["open_year"].isna() | (df["open_year"] <= snapshot_year)
    close_ok = df["close_year"].isna() | (df["close_year"] > snapshot_year)
    return open_ok & close_ok


def safe_pct_change(newer: pd.Series, older: pd.Series) -> pd.Series:
    diff = newer - older
    pct = np.where(older > 0, (diff / older) * 100, np.nan)
    return pd.Series(pct, index=older.index)


def build_snapshot_gdf(df: pd.DataFrame, snapshot_year: int) -> gpd.GeoDataFrame:
    active = df[is_active_in_year(df, snapshot_year)].copy()

    geometry = gpd.points_from_xy(active["long_wgs84"], active["lat_wgs84"], crs="EPSG:4326")
    gdf = gpd.GeoDataFrame(active, geometry=geometry, crs="EPSG:4326").to_crs("EPSG:27700")

    gdf = gdf.drop_duplicates(subset=["id"]).copy()
    gdf["snapshot_year"] = snapshot_year
    return gdf


def deduplicate_nearest_results(
    nearest_gdf: gpd.GeoDataFrame,
    zone_key: str,
    distance_col: str,
    bank_id_col: str,
) -> pd.DataFrame:
    """
    sjoin_nearest can return multiple rows per zone when there are ties.
    Keep one deterministic record per zone:
    1. smallest distance
    2. smallest bank id
    """
    df = pd.DataFrame(nearest_gdf.drop(columns="geometry")).copy()

    df[distance_col] = pd.to_numeric(df[distance_col], errors="coerce")
    df[bank_id_col] = pd.to_numeric(df[bank_id_col], errors="coerce")

    before_rows = len(df)
    duplicate_zone_count = int(df.duplicated(subset=[zone_key]).sum())

    df = df.sort_values(
        by=[zone_key, distance_col, bank_id_col],
        ascending=[True, True, True],
        na_position="last",
    ).drop_duplicates(subset=[zone_key], keep="first").reset_index(drop=True)

    after_rows = len(df)

    print(f"\nDedup summary for {distance_col}:")
    print(f"Rows before deduplication: {before_rows}")
    print(f"Duplicate zone rows detected: {duplicate_zone_count}")
    print(f"Rows after deduplication: {after_rows}")

    return df


def main() -> None:
    print("Starting temporal bank accessibility build...")

    data_dir = PROCESSED_DIR.parent
    raw_dir = data_dir / "raw"
    processed_bank_dir = PROCESSED_DIR / "bank_closures"
    processed_access_dir = PROCESSED_DIR / "accessibility"
    processed_access_dir.mkdir(parents=True, exist_ok=True)

    geolytix_dir = resolve_geolytix_folder(raw_dir)
    bank_csv_path = resolve_required_file(geolytix_dir, "geolytix_uk_open_bank_branches.csv")

    origins_path = processed_access_dir / "zone_origins_2022.gpkg"
    closure_change_path = processed_bank_dir / "bank_closure_change_features_2022.csv"

    print(f"Geolytix source folder: {geolytix_dir}")
    print(f"Bank CSV path: {bank_csv_path}")
    print(f"Zone origins path: {origins_path}")
    print(f"Closure change path: {closure_change_path}")

    print("\nLoading Geolytix bank universe...")
    banks_raw = pd.read_csv(bank_csv_path)

    print(f"Rows loaded from bank CSV: {len(banks_raw)}")
    print(f"Columns loaded: {banks_raw.columns.tolist()}")

    missing_expected = [col for col in EXPECTED_COLUMNS if col not in banks_raw.columns]
    if missing_expected:
        raise ValueError(f"Missing expected columns in bank CSV: {missing_expected}")

    banks = banks_raw[EXPECTED_COLUMNS].copy()
    banks = clean_text_columns(
        banks,
        [
            "brand_full",
            "brand_short",
            "branch_name",
            "branch_type",
            "add_one",
            "add_two",
            "suburb",
            "town",
            "region",
            "postcode",
            "status",
        ],
    )

    banks["region"] = banks["region"].fillna("").str.strip()
    banks["status"] = banks["status"].fillna("").str.strip()
    banks["long_wgs84"] = pd.to_numeric(banks["long_wgs84"], errors="coerce")
    banks["lat_wgs84"] = pd.to_numeric(banks["lat_wgs84"], errors="coerce")
    banks["close_year"] = pd.to_numeric(banks["close_year"], errors="coerce")
    banks["open_year"] = pd.to_numeric(banks["open_year"], errors="coerce")
    banks["po_dist"] = pd.to_numeric(banks["po_dist"], errors="coerce")

    print("\nFiltering to Scotland rows only...")
    banks = banks[banks["region"].str.lower() == "scotland"].copy()

    print("Dropping rows with missing coordinates...")
    before_drop = len(banks)
    banks = banks.dropna(subset=["long_wgs84", "lat_wgs84"]).copy()
    dropped_missing_coords = before_drop - len(banks)

    print(f"Scotland bank rows after cleaning: {len(banks)}")
    print(f"Dropped missing-coordinate rows: {dropped_missing_coords}")

    print("\nBuilding 2019 and 2023 active bank snapshots...")
    banks_2019 = build_snapshot_gdf(banks, PRE_YEAR)
    banks_2023 = build_snapshot_gdf(banks, POST_YEAR)

    print(f"Active bank rows in {PRE_YEAR}: {len(banks_2019)}")
    print(f"Active bank rows in {POST_YEAR}: {len(banks_2023)}")

    print("\nLoading zone origins and closure change features...")
    zone_origins = gpd.read_file(origins_path, layer="zone_origins_2022")
    closure_change = pd.read_csv(closure_change_path)

    print(f"Zone origins rows loaded: {len(zone_origins)}")
    print(f"Closure change rows loaded: {len(closure_change)}")

    if zone_origins.crs != banks_2019.crs:
        zone_origins = zone_origins.to_crs(banks_2019.crs)

    origin_keep_cols = [
        "dz_code_2022",
        "dz_name_2022",
        "ur6_name",
        "ur8_name",
        "is_rural",
        "is_accessible_rural",
        "is_remote_rural",
        "origin_x",
        "origin_y",
        "geometry",
    ]
    origin_keep_cols = [col for col in origin_keep_cols if col in zone_origins.columns]
    zone_origins = zone_origins[origin_keep_cols].copy()

    print(f"\nComputing nearest-bank accessibility for {PRE_YEAR}...")
    nearest_2019 = gpd.sjoin_nearest(
        zone_origins,
        banks_2019[["id", "brand_full", "branch_type", "town", "geometry"]],
        how="left",
        distance_col=f"dist_to_nearest_bank_m_{PRE_YEAR}",
    )

    nearest_2019 = nearest_2019.rename(
        columns={
            "id": f"nearest_bank_id_{PRE_YEAR}",
            "brand_full": f"nearest_bank_brand_{PRE_YEAR}",
            "branch_type": f"nearest_bank_type_{PRE_YEAR}",
            "town": f"nearest_bank_town_{PRE_YEAR}",
        }
    ).drop(columns=["index_right"], errors="ignore")

    nearest_2019 = deduplicate_nearest_results(
        nearest_2019,
        zone_key="dz_code_2022",
        distance_col=f"dist_to_nearest_bank_m_{PRE_YEAR}",
        bank_id_col=f"nearest_bank_id_{PRE_YEAR}",
    )

    print(f"Computing nearest-bank accessibility for {POST_YEAR}...")
    nearest_2023 = gpd.sjoin_nearest(
        zone_origins,
        banks_2023[["id", "brand_full", "branch_type", "town", "geometry"]],
        how="left",
        distance_col=f"dist_to_nearest_bank_m_{POST_YEAR}",
    )

    nearest_2023 = nearest_2023.rename(
        columns={
            "id": f"nearest_bank_id_{POST_YEAR}",
            "brand_full": f"nearest_bank_brand_{POST_YEAR}",
            "branch_type": f"nearest_bank_type_{POST_YEAR}",
            "town": f"nearest_bank_town_{POST_YEAR}",
        }
    ).drop(columns=["index_right"], errors="ignore")

    nearest_2023 = deduplicate_nearest_results(
        nearest_2023,
        zone_key="dz_code_2022",
        distance_col=f"dist_to_nearest_bank_m_{POST_YEAR}",
        bank_id_col=f"nearest_bank_id_{POST_YEAR}",
    )

    temporal = nearest_2019.merge(
        nearest_2023[
            [
                "dz_code_2022",
                f"nearest_bank_id_{POST_YEAR}",
                f"nearest_bank_brand_{POST_YEAR}",
                f"nearest_bank_type_{POST_YEAR}",
                f"nearest_bank_town_{POST_YEAR}",
                f"dist_to_nearest_bank_m_{POST_YEAR}",
            ]
        ],
        on="dz_code_2022",
        how="inner",
        validate="1:1",
    )

    temporal[f"dist_to_nearest_bank_km_{PRE_YEAR}"] = temporal[f"dist_to_nearest_bank_m_{PRE_YEAR}"] / 1000.0
    temporal[f"dist_to_nearest_bank_km_{POST_YEAR}"] = temporal[f"dist_to_nearest_bank_m_{POST_YEAR}"] / 1000.0

    temporal["bank_distance_change_km_post_minus_pre"] = (
        temporal[f"dist_to_nearest_bank_km_{POST_YEAR}"]
        - temporal[f"dist_to_nearest_bank_km_{PRE_YEAR}"]
    )

    temporal["bank_distance_change_m_post_minus_pre"] = (
        temporal[f"dist_to_nearest_bank_m_{POST_YEAR}"]
        - temporal[f"dist_to_nearest_bank_m_{PRE_YEAR}"]
    )

    temporal["bank_distance_pct_change_post_vs_pre"] = safe_pct_change(
        temporal[f"dist_to_nearest_bank_km_{POST_YEAR}"],
        temporal[f"dist_to_nearest_bank_km_{PRE_YEAR}"],
    )

    temporal["bank_access_deterioration_flag"] = (
        temporal["bank_distance_change_km_post_minus_pre"] > 0
    ).astype(int)

    temporal["bank_access_improvement_flag"] = (
        temporal["bank_distance_change_km_post_minus_pre"] < 0
    ).astype(int)

    temporal["bank_access_no_change_flag"] = (
        temporal["bank_distance_change_km_post_minus_pre"].round(6) == 0
    ).astype(int)

    temporal["bank_access_major_deterioration_flag"] = (
        temporal["bank_distance_change_km_post_minus_pre"] >= 1.0
    ).astype(int)

    temporal["bank_access_severe_deterioration_flag"] = (
        temporal["bank_distance_change_km_post_minus_pre"] >= 5.0
    ).astype(int)

    closure_keep_cols = [
        "dz_code_2022",
        "pre_covid_closures_total",
        "post_covid_closures_total",
        "pre_covid_closures_annual_rate",
        "post_covid_closures_annual_rate",
        "closure_rate_change_post_minus_pre",
        "closure_count_change_post_minus_pre",
        "any_pre_covid_closure_flag",
        "any_post_covid_closure_flag",
        "post_covid_only_closure_flag",
        "closure_deterioration_flag",
        "closure_persistence_flag",
        "first_closure_year",
        "cumulative_closures_to_2023",
    ]
    closure_keep_cols = [col for col in closure_keep_cols if col in closure_change.columns]

    temporal = temporal.merge(
        closure_change[closure_keep_cols],
        on="dz_code_2022",
        how="left",
        validate="1:1",
    )

    temporal = temporal.sort_values(
        by=[
            "bank_distance_change_km_post_minus_pre",
            "post_covid_closures_total",
            "cumulative_closures_to_2023",
        ],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    temporal["bank_access_worsening_rank"] = temporal.index + 1

    summary_by_ur6 = (
        temporal.groupby("ur6_name", dropna=False)
        .agg(
            zone_count=("dz_code_2022", "count"),
            mean_bank_km_2019=(f"dist_to_nearest_bank_km_{PRE_YEAR}", "mean"),
            mean_bank_km_2023=(f"dist_to_nearest_bank_km_{POST_YEAR}", "mean"),
            mean_bank_change_km=("bank_distance_change_km_post_minus_pre", "mean"),
            deteriorating_zone_count=("bank_access_deterioration_flag", "sum"),
            major_deterioration_zone_count=("bank_access_major_deterioration_flag", "sum"),
            severe_deterioration_zone_count=("bank_access_severe_deterioration_flag", "sum"),
        )
        .reset_index()
        .sort_values("mean_bank_change_km", ascending=False)
        .reset_index(drop=True)
    )

    rural_summary = (
        temporal[temporal["is_rural"] == 1]
        .groupby("ur6_name", dropna=False)
        .agg(
            rural_zone_count=("dz_code_2022", "count"),
            mean_bank_km_2019=(f"dist_to_nearest_bank_km_{PRE_YEAR}", "mean"),
            mean_bank_km_2023=(f"dist_to_nearest_bank_km_{POST_YEAR}", "mean"),
            mean_bank_change_km=("bank_distance_change_km_post_minus_pre", "mean"),
            deteriorating_zone_count=("bank_access_deterioration_flag", "sum"),
            major_deterioration_zone_count=("bank_access_major_deterioration_flag", "sum"),
            severe_deterioration_zone_count=("bank_access_severe_deterioration_flag", "sum"),
        )
        .reset_index()
        .sort_values("mean_bank_change_km", ascending=False)
        .reset_index(drop=True)
    )

    top_100_worsening_rural = (
        temporal[temporal["is_rural"] == 1]
        .copy()
        .sort_values(
            by=[
                "bank_distance_change_km_post_minus_pre",
                "post_covid_closures_total",
                "cumulative_closures_to_2023",
            ],
            ascending=[False, False, False],
        )
        .reset_index(drop=True)
    )

    top_100_worsening_rural["rural_bank_access_worsening_rank"] = top_100_worsening_rural.index + 1

    top_100_worsening_rural = top_100_worsening_rural[
        [
            "rural_bank_access_worsening_rank",
            "dz_code_2022",
            "dz_name_2022",
            "ur6_name",
            "ur8_name",
            f"dist_to_nearest_bank_km_{PRE_YEAR}",
            f"dist_to_nearest_bank_km_{POST_YEAR}",
            "bank_distance_change_km_post_minus_pre",
            "bank_distance_pct_change_post_vs_pre",
            "post_covid_closures_total",
            "closure_rate_change_post_minus_pre",
            "bank_access_deterioration_flag",
            "bank_access_major_deterioration_flag",
            "bank_access_severe_deterioration_flag",
        ]
    ].head(100)

    overall_summary = pd.DataFrame(
        {
            "metric": [
                "total_zones",
                "rural_zones",
                "mean_bank_km_2019",
                "mean_bank_km_2023",
                "zones_with_bank_access_deterioration",
                "zones_with_major_bank_access_deterioration",
                "zones_with_severe_bank_access_deterioration",
            ],
            "value": [
                int(len(temporal)),
                int((temporal["is_rural"] == 1).sum()),
                float(temporal[f"dist_to_nearest_bank_km_{PRE_YEAR}"].mean()),
                float(temporal[f"dist_to_nearest_bank_km_{POST_YEAR}"].mean()),
                int(temporal["bank_access_deterioration_flag"].sum()),
                int(temporal["bank_access_major_deterioration_flag"].sum()),
                int(temporal["bank_access_severe_deterioration_flag"].sum()),
            ],
        }
    )

    print("\n--- Temporal Bank Accessibility Summary ---")
    print(f"Temporal accessibility rows: {len(temporal)}")
    print(f"Summary by UR6 rows: {len(summary_by_ur6)}")
    print(f"Rural summary rows: {len(rural_summary)}")
    print(f"Top 100 worsening rural rows: {len(top_100_worsening_rural)}")

    print("\nOverall summary:")
    print(overall_summary)

    print("\nSummary by UR6:")
    print(summary_by_ur6)

    print("\nRural summary:")
    print(rural_summary)

    print("\nTop 10 worsening rural zones:")
    print(top_100_worsening_rural.head(10))

    snapshot_2019_gpkg = processed_access_dir / "banks_snapshot_2019_scotland.gpkg"
    snapshot_2023_gpkg = processed_access_dir / "banks_snapshot_2023_scotland.gpkg"
    temporal_csv = processed_access_dir / "bank_accessibility_temporal_2019_2023.csv"
    summary_ur6_csv = processed_access_dir / "bank_accessibility_temporal_summary_by_ur6.csv"
    rural_summary_csv = processed_access_dir / "bank_accessibility_temporal_rural_summary.csv"
    overall_summary_csv = processed_access_dir / "bank_accessibility_temporal_overall_summary.csv"
    top100_csv = processed_access_dir / "bank_accessibility_top_100_worsening_rural_zones.csv"

    banks_2019.to_file(
        snapshot_2019_gpkg,
        layer="banks_snapshot_2019_scotland",
        driver="GPKG",
    )
    banks_2023.to_file(
        snapshot_2023_gpkg,
        layer="banks_snapshot_2023_scotland",
        driver="GPKG",
    )

    temporal.to_csv(temporal_csv, index=False)
    summary_by_ur6.to_csv(summary_ur6_csv, index=False)
    rural_summary.to_csv(rural_summary_csv, index=False)
    overall_summary.to_csv(overall_summary_csv, index=False)
    top_100_worsening_rural.to_csv(top100_csv, index=False)

    print("\nTemporal bank accessibility build completed successfully.")
    print(f"2019 bank snapshot GPKG: {snapshot_2019_gpkg}")
    print(f"2023 bank snapshot GPKG: {snapshot_2023_gpkg}")
    print(f"Temporal accessibility CSV: {temporal_csv}")
    print(f"Summary by UR6 CSV: {summary_ur6_csv}")
    print(f"Rural summary CSV: {rural_summary_csv}")
    print(f"Overall summary CSV: {overall_summary_csv}")
    print(f"Top 100 worsening rural CSV: {top100_csv}")


if __name__ == "__main__":
    main()