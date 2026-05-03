from pathlib import Path
import geopandas as gpd
import pandas as pd

from config.paths import (
    DZ_BOUNDARIES_2011_DIR,
    DZ_BOUNDARIES_2022_DIR,
    RURAL_CLASSIFICATION_FILE,
    PROCESSED_DIR,
)


def find_shapefile(folder: Path) -> Path:
    """
    Find the first .shp file inside a folder or its subfolders.
    """
    shapefiles = list(folder.rglob("*.shp"))
    if not shapefiles:
        raise FileNotFoundError(f"No shapefile found inside: {folder}")
    return shapefiles[0]


def read_csv_robust(file_path: Path) -> pd.DataFrame:
    """
    Read a CSV file using a small fallback list of encodings.
    """
    encodings_to_try = ["utf-8", "cp1252", "latin1"]

    for encoding in encodings_to_try:
        try:
            print(f"Trying to read CSV with encoding: {encoding}")
            return pd.read_csv(file_path, encoding=encoding)
        except UnicodeDecodeError:
            print(f"Failed with encoding: {encoding}")

    raise UnicodeDecodeError(
        "read_csv",
        b"",
        0,
        1,
        f"Unable to decode CSV file with tried encodings: {encodings_to_try}",
    )


def prepare_2011_zones(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Standardise the 2011 Data Zone layer into a clean support geography table.
    """
    gdf = gdf.copy()

    gdf = gdf.rename(
        columns={
            "DataZone": "dz_code_2011",
            "Name": "dz_name_2011",
        }
    )

    gdf["area_sqkm"] = gdf.geometry.area / 1_000_000
    centroids = gdf.geometry.centroid
    gdf["centroid_x"] = centroids.x
    gdf["centroid_y"] = centroids.y

    keep_cols = [
        "dz_code_2011",
        "dz_name_2011",
        "area_sqkm",
        "centroid_x",
        "centroid_y",
        "geometry",
    ]

    gdf = gdf[keep_cols].copy()

    if gdf["dz_code_2011"].duplicated().any():
        raise ValueError("Duplicate dz_code_2011 values found in 2011 geography.")

    return gdf


def prepare_2022_zones(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Standardise the 2022 Data Zone layer into a clean reporting geography table.
    """
    gdf = gdf.copy()

    gdf = gdf.rename(
        columns={
            "dzcode": "dz_code_2022",
            "dzname": "dz_name_2022",
        }
    )

    gdf["area_sqkm"] = gdf.geometry.area / 1_000_000
    centroids = gdf.geometry.centroid
    gdf["centroid_x"] = centroids.x
    gdf["centroid_y"] = centroids.y

    keep_cols = [
        "dz_code_2022",
        "dz_name_2022",
        "area_sqkm",
        "centroid_x",
        "centroid_y",
        "geometry",
    ]

    gdf = gdf[keep_cols].copy()

    if gdf["dz_code_2022"].duplicated().any():
        raise ValueError("Duplicate dz_code_2022 values found in 2022 geography.")

    return gdf


def prepare_rural_lookup(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise the rural classification lookup for joining to 2022 Data Zones.
    """
    df = df.copy()

    df = df.rename(
        columns={
            "DZ22_Code": "dz_code_2022",
            "DZ22_Name": "dz_name_lookup_2022",
            "UR2_Code": "ur2_code",
            "UR2_Name": "ur2_name",
            "UR3_Code": "ur3_code",
            "UR3_Name": "ur3_name",
            "UR6_Code": "ur6_code",
            "UR6_Name": "ur6_name",
            "UR8_Code": "ur8_code",
            "UR8_Name": "ur8_name",
        }
    )

    keep_cols = [
        "dz_code_2022",
        "dz_name_lookup_2022",
        "ur2_code",
        "ur2_name",
        "ur3_code",
        "ur3_name",
        "ur6_code",
        "ur6_name",
        "ur8_code",
        "ur8_name",
    ]

    df = df[keep_cols].copy()

    if df["dz_code_2022"].duplicated().any():
        raise ValueError("Duplicate dz_code_2022 values found in rural lookup.")

    return df


def join_rural_to_2022_zones(
    zones_master_2022: gpd.GeoDataFrame,
    rural_lookup: pd.DataFrame,
) -> gpd.GeoDataFrame:
    """
    Join rural classification lookup to the 2022 master geography.
    """
    merged = zones_master_2022.merge(
        rural_lookup,
        on="dz_code_2022",
        how="left",
        validate="1:1",
    )

    missing_matches = merged["ur6_code"].isna().sum()
    if missing_matches > 0:
        raise ValueError(
            f"Rural join failed for {missing_matches} 2022 Data Zones."
        )

    return merged


def add_rural_flags(zones_master_2022: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Add dissertation-ready rural flags using UR6 as the primary filter rule.
    """
    gdf = zones_master_2022.copy()

    gdf["is_accessible_rural"] = (gdf["ur6_name"] == "Accessible Rural").astype(int)
    gdf["is_remote_rural"] = (gdf["ur6_name"] == "Remote Rural").astype(int)
    gdf["is_rural"] = gdf["ur6_name"].isin(
        ["Accessible Rural", "Remote Rural"]
    ).astype(int)

    return gdf


def main() -> None:
    print("Starting geography preparation build...")

    geography_dir = PROCESSED_DIR / "geography"
    geography_dir.mkdir(parents=True, exist_ok=True)

    shp_2011 = find_shapefile(DZ_BOUNDARIES_2011_DIR)
    shp_2022 = find_shapefile(DZ_BOUNDARIES_2022_DIR)

    print(f"2011 shapefile found: {shp_2011}")
    print(f"2022 shapefile found: {shp_2022}")
    print(f"Rural classification file found: {RURAL_CLASSIFICATION_FILE}")

    gdf_2011_raw = gpd.read_file(shp_2011)
    gdf_2022_raw = gpd.read_file(shp_2022)

    print("\nPreparing clean 2011 support geography...")
    zones_support_2011 = prepare_2011_zones(gdf_2011_raw)

    print("Preparing clean 2022 reporting geography...")
    zones_master_2022 = prepare_2022_zones(gdf_2022_raw)

    print("\nInspecting and preparing rural classification lookup...")
    rural_df_raw = read_csv_robust(RURAL_CLASSIFICATION_FILE)
    rural_lookup = prepare_rural_lookup(rural_df_raw)

    print("Joining rural classification to 2022 master geography...")
    zones_master_2022 = join_rural_to_2022_zones(zones_master_2022, rural_lookup)

    print("Adding dissertation rural flags...")
    zones_master_2022 = add_rural_flags(zones_master_2022)

    output_2011 = geography_dir / "zones_support_2011.gpkg"
    output_2022 = geography_dir / "zones_master_2022.gpkg"

    zones_support_2011.to_file(output_2011, layer="zones_support_2011", driver="GPKG")
    zones_master_2022.to_file(output_2022, layer="zones_master_2022", driver="GPKG")

    print("\n--- Build Summary ---")
    print(f"2011 rows: {len(zones_support_2011)}")
    print(f"2022 rows: {len(zones_master_2022)}")
    print(f"2011 CRS: {zones_support_2011.crs}")
    print(f"2022 CRS: {zones_master_2022.crs}")
    print(f"2011 output: {output_2011}")
    print(f"2022 output: {output_2022}")

    print("\n2022 columns after rural join and flags:")
    print(zones_master_2022.columns.tolist())

    print("\nUR6 value counts:")
    print(zones_master_2022["ur6_name"].value_counts(dropna=False))

    print("\nRural flag counts:")
    print(zones_master_2022["is_rural"].value_counts(dropna=False))

    print("\nAccessible rural flag counts:")
    print(zones_master_2022["is_accessible_rural"].value_counts(dropna=False))

    print("\nRemote rural flag counts:")
    print(zones_master_2022["is_remote_rural"].value_counts(dropna=False))

    print("\nGeography preparation build completed successfully.")


if __name__ == "__main__":
    main()