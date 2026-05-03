from pathlib import Path
import pandas as pd
import geopandas as gpd

from config.paths import PROCESSED_DIR


def compute_nearest_distance_by_service(
    origins: gpd.GeoDataFrame,
    destinations: gpd.GeoDataFrame,
    service_type: str,
    output_col: str,
) -> pd.DataFrame:
    """
    Compute nearest straight-line distance from each origin to the nearest
    destination of a given service type.
    """
    subset = destinations[destinations["service_type"] == service_type].copy()

    if subset.empty:
        raise ValueError(f"No destination points found for service_type={service_type}")

    joined = gpd.sjoin_nearest(
        origins[["dz_code_2022", "geometry"]].copy(),
        subset[["geometry"]].copy(),
        how="left",
        distance_col=output_col,
    )

    result = joined[["dz_code_2022", output_col]].copy()
    result = result.drop_duplicates(subset=["dz_code_2022"])

    return result


def compute_nearest_distance_any_service(
    origins: gpd.GeoDataFrame,
    destinations: gpd.GeoDataFrame,
    output_col: str,
) -> pd.DataFrame:
    """
    Compute nearest straight-line distance from each origin to the nearest
    destination across all service types.
    """
    joined = gpd.sjoin_nearest(
        origins[["dz_code_2022", "geometry"]].copy(),
        destinations[["geometry"]].copy(),
        how="left",
        distance_col=output_col,
    )

    result = joined[["dz_code_2022", output_col]].copy()
    result = result.drop_duplicates(subset=["dz_code_2022"])

    return result


def main() -> None:
    print("Starting baseline accessibility build...")

    accessibility_dir = PROCESSED_DIR / "accessibility"

    origins_path = accessibility_dir / "zone_origins_2022.gpkg"
    destinations_path = accessibility_dir / "service_destinations_current.gpkg"
    context_path = PROCESSED_DIR / "context" / "zone_year_context_2022.csv"

    print(f"Loading origins from: {origins_path}")
    origins = gpd.read_file(origins_path, layer="zone_origins_2022")

    print(f"Loading destinations from: {destinations_path}")
    destinations = gpd.read_file(destinations_path, layer="service_destinations_current")

    print(f"Loading context from: {context_path}")
    context_df = pd.read_csv(context_path)

    print(f"Origins rows: {len(origins)}")
    print(f"Destinations rows: {len(destinations)}")
    print(f"Context rows: {len(context_df)}")

    if origins.crs != destinations.crs:
        destinations = destinations.to_crs(origins.crs)

    print("\nComputing nearest bank distance...")
    bank_dist = compute_nearest_distance_by_service(
        origins, destinations, "bank", "dist_to_nearest_bank_m"
    )

    print("Computing nearest ATM distance...")
    atm_dist = compute_nearest_distance_by_service(
        origins, destinations, "atm", "dist_to_nearest_atm_m"
    )

    print("Computing nearest post office distance...")
    post_dist = compute_nearest_distance_by_service(
        origins, destinations, "post_office", "dist_to_nearest_post_office_m"
    )

    print("Computing nearest financial access point distance (all services)...")
    any_dist = compute_nearest_distance_any_service(
        origins, destinations, "dist_to_nearest_any_access_point_m"
    )

    zone_access_baseline = origins.drop(columns="geometry").copy()

    zone_access_baseline = zone_access_baseline.merge(
        bank_dist, on="dz_code_2022", how="left", validate="1:1"
    )
    zone_access_baseline = zone_access_baseline.merge(
        atm_dist, on="dz_code_2022", how="left", validate="1:1"
    )
    zone_access_baseline = zone_access_baseline.merge(
        post_dist, on="dz_code_2022", how="left", validate="1:1"
    )
    zone_access_baseline = zone_access_baseline.merge(
        any_dist, on="dz_code_2022", how="left", validate="1:1"
    )

    for col in [
        "dist_to_nearest_bank_m",
        "dist_to_nearest_atm_m",
        "dist_to_nearest_post_office_m",
        "dist_to_nearest_any_access_point_m",
    ]:
        km_col = col.replace("_m", "_km")
        zone_access_baseline[km_col] = zone_access_baseline[col] / 1000

    output_zone_csv = accessibility_dir / "zone_accessibility_baseline_2022.csv"
    zone_access_baseline.to_csv(output_zone_csv, index=False)

    context_access = context_df.merge(
        zone_access_baseline,
        on="dz_code_2022",
        how="left",
        validate="m:1",
    )

    output_context_csv = accessibility_dir / "zone_year_accessibility_baseline_2022.csv"
    context_access.to_csv(output_context_csv, index=False)

    print("\n--- Baseline Accessibility Summary ---")
    print(f"Zone accessibility rows: {len(zone_access_baseline)}")
    print(f"Unique zones: {zone_access_baseline['dz_code_2022'].nunique()}")

    print("\nMissing values by distance column:")
    print(
        zone_access_baseline[
            [
                "dist_to_nearest_bank_m",
                "dist_to_nearest_atm_m",
                "dist_to_nearest_post_office_m",
                "dist_to_nearest_any_access_point_m",
            ]
        ].isna().sum()
    )

    print("\nDistance summary (meters):")
    print(
        zone_access_baseline[
            [
                "dist_to_nearest_bank_m",
                "dist_to_nearest_atm_m",
                "dist_to_nearest_post_office_m",
                "dist_to_nearest_any_access_point_m",
            ]
        ].describe()
    )

    print("\nRows by rural flag:")
    print(zone_access_baseline["is_rural"].value_counts(dropna=False))

    print("\nBaseline accessibility build completed successfully.")
    print(f"Zone accessibility output: {output_zone_csv}")
    print(f"Zone-year accessibility output: {output_context_csv}")


if __name__ == "__main__":
    main()