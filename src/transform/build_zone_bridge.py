from pathlib import Path
import geopandas as gpd
import pandas as pd

from config.paths import PROCESSED_DIR, INTERIM_DIR


def main() -> None:
    print("Starting 2011-to-2022 zone bridge build...")

    geography_dir = PROCESSED_DIR / "geography"
    bridge_dir = INTERIM_DIR / "geography_bridge"
    bridge_dir.mkdir(parents=True, exist_ok=True)

    zones_2011_path = geography_dir / "zones_support_2011.gpkg"
    zones_2022_path = geography_dir / "zones_master_2022.gpkg"

    print(f"Loading 2011 support geography: {zones_2011_path}")
    print(f"Loading 2022 master geography: {zones_2022_path}")

    zones_2011 = gpd.read_file(zones_2011_path, layer="zones_support_2011")
    zones_2022 = gpd.read_file(zones_2022_path, layer="zones_master_2022")

    print(f"2011 rows loaded: {len(zones_2011)}")
    print(f"2022 rows loaded: {len(zones_2022)}")
    print(f"2011 CRS: {zones_2011.crs}")
    print(f"2022 CRS: {zones_2022.crs}")

    if zones_2011.crs != zones_2022.crs:
        raise ValueError("CRS mismatch between 2011 and 2022 geography layers.")

    zones_2011 = zones_2011[
        ["dz_code_2011", "dz_name_2011", "area_sqkm", "geometry"]
    ].copy()
    zones_2022 = zones_2022[
        ["dz_code_2022", "dz_name_2022", "area_sqkm", "is_rural", "geometry"]
    ].copy()

    zones_2011 = zones_2011.rename(columns={"area_sqkm": "area_2011_sqkm"})
    zones_2022 = zones_2022.rename(columns={"area_sqkm": "area_2022_sqkm"})

    print("\nCreating spatial intersection bridge...")
    bridge = gpd.overlay(zones_2011, zones_2022, how="intersection")

    bridge["intersection_sqkm"] = bridge.geometry.area / 1_000_000
    bridge = bridge[bridge["intersection_sqkm"] > 0].copy()

    bridge["pct_of_2011"] = bridge["intersection_sqkm"] / bridge["area_2011_sqkm"]
    bridge["pct_of_2022"] = bridge["intersection_sqkm"] / bridge["area_2022_sqkm"]

    bridge = bridge.sort_values(
        ["dz_code_2011", "pct_of_2011"],
        ascending=[True, False]
    ).copy()
    bridge["rank_within_2011"] = bridge.groupby("dz_code_2011").cumcount() + 1

    bridge = bridge.sort_values(
        ["dz_code_2022", "pct_of_2022"],
        ascending=[True, False]
    ).copy()
    bridge["rank_within_2022"] = bridge.groupby("dz_code_2022").cumcount() + 1

    bridge["is_primary_match_for_2011"] = (bridge["rank_within_2011"] == 1).astype(int)
    bridge["is_primary_match_for_2022"] = (bridge["rank_within_2022"] == 1).astype(int)

    bridge_gpkg_path = bridge_dir / "zone_lookup_bridge.gpkg"
    bridge_csv_path = bridge_dir / "zone_lookup_bridge.csv"
    qa_csv_path = bridge_dir / "zone_lookup_bridge_qa_summary.csv"

    bridge.to_file(bridge_gpkg_path, layer="zone_lookup_bridge", driver="GPKG")

    bridge_no_geom = pd.DataFrame(bridge.drop(columns="geometry"))
    bridge_no_geom.to_csv(bridge_csv_path, index=False)

    primary_2011 = bridge[bridge["is_primary_match_for_2011"] == 1].copy()
    primary_2022 = bridge[bridge["is_primary_match_for_2022"] == 1].copy()

    weak_2011 = (primary_2011["pct_of_2011"] < 0.50).sum()
    weak_2022 = (primary_2022["pct_of_2022"] < 0.50).sum()

    split_counts_2011 = bridge.groupby("dz_code_2011").size()
    split_counts_2022 = bridge.groupby("dz_code_2022").size()

    qa_summary = pd.DataFrame(
        {
            "metric": [
                "bridge_rows",
                "unique_2011_zones",
                "unique_2022_zones",
                "primary_matches_2011",
                "primary_matches_2022",
                "weak_primary_2011_lt_50pct",
                "weak_primary_2022_lt_50pct",
                "mean_links_per_2011_zone",
                "mean_links_per_2022_zone",
                "max_links_for_single_2011_zone",
                "max_links_for_single_2022_zone",
            ],
            "value": [
                len(bridge),
                bridge["dz_code_2011"].nunique(),
                bridge["dz_code_2022"].nunique(),
                primary_2011["dz_code_2011"].nunique(),
                primary_2022["dz_code_2022"].nunique(),
                int(weak_2011),
                int(weak_2022),
                round(split_counts_2011.mean(), 4),
                round(split_counts_2022.mean(), 4),
                int(split_counts_2011.max()),
                int(split_counts_2022.max()),
            ],
        }
    )
    qa_summary.to_csv(qa_csv_path, index=False)

    print("\n--- Bridge Build Summary ---")
    print(f"Bridge rows: {len(bridge)}")
    print(f"Unique 2011 zones in bridge: {bridge['dz_code_2011'].nunique()}")
    print(f"Unique 2022 zones in bridge: {bridge['dz_code_2022'].nunique()}")

    print("\nPrimary-match coverage:")
    print(f"Primary matches for 2011 zones: {primary_2011['dz_code_2011'].nunique()}")
    print(f"Primary matches for 2022 zones: {primary_2022['dz_code_2022'].nunique()}")

    print("\nWeak primary-match checks (< 50% overlap):")
    print(f"Weak primary 2011 matches: {weak_2011}")
    print(f"Weak primary 2022 matches: {weak_2022}")

    print("\nAverage split complexity:")
    print(f"Mean links per 2011 zone: {split_counts_2011.mean():.4f}")
    print(f"Mean links per 2022 zone: {split_counts_2022.mean():.4f}")
    print(f"Max links for a single 2011 zone: {split_counts_2011.max()}")
    print(f"Max links for a single 2022 zone: {split_counts_2022.max()}")

    print("\nTop 10 weakest primary 2011 matches:")
    weakest_2011_cols = [
        "dz_code_2011",
        "dz_name_2011",
        "dz_code_2022",
        "dz_name_2022",
        "pct_of_2011",
        "pct_of_2022",
        "intersection_sqkm",
        "is_rural",
    ]
    print(
        primary_2011[weakest_2011_cols]
        .sort_values("pct_of_2011", ascending=True)
        .head(10)
    )

    print("\nTop 10 weakest primary 2022 matches:")
    weakest_2022_cols = [
        "dz_code_2011",
        "dz_name_2011",
        "dz_code_2022",
        "dz_name_2022",
        "pct_of_2011",
        "pct_of_2022",
        "intersection_sqkm",
        "is_rural",
    ]
    print(
        primary_2022[weakest_2022_cols]
        .sort_values("pct_of_2022", ascending=True)
        .head(10)
    )

    print(f"\nBridge GeoPackage output: {bridge_gpkg_path}")
    print(f"Bridge CSV output: {bridge_csv_path}")
    print(f"Bridge QA summary output: {qa_csv_path}")
    print("\n2011-to-2022 zone bridge build completed successfully.")


if __name__ == "__main__":
    main()