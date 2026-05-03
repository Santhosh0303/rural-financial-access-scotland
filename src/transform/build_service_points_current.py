from pathlib import Path
import pandas as pd
import geopandas as gpd

from config.paths import INTERIM_DIR, PROCESSED_DIR


def load_service_layer(file_path: Path, layer_name: str, expected_service_type: str) -> gpd.GeoDataFrame:
    """
    Load one extracted raw service layer and enforce the expected service_type.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Missing service file: {file_path}")

    gdf = gpd.read_file(file_path, layer=layer_name)

    if gdf.empty:
        print(f"Warning: {expected_service_type} layer is empty.")
        return gdf

    gdf = gdf.copy()
    gdf["service_type"] = expected_service_type

    return gdf


def convert_services_to_points(gdf: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, dict]:
    """
    Convert service geometries into point-based representations.

    Rules:
    - Point: keep as is
    - MultiPoint: explode into individual points
    - Polygon / MultiPolygon: convert to representative_point()
    - Anything else: drop
    """
    out = gdf.copy()
    out["source_geometry_type"] = out.geometry.geom_type

    summary = {
        "original_rows": len(out),
        "point_rows_kept": 0,
        "multipoint_rows_expanded": 0,
        "polygon_rows_converted": 0,
        "unsupported_rows_dropped": 0,
    }

    # Keep points
    points = out[out.geometry.geom_type == "Point"].copy()
    summary["point_rows_kept"] = len(points)

    # Expand multipoints into separate point rows
    multipoints = out[out.geometry.geom_type == "MultiPoint"].copy()
    if not multipoints.empty:
        multipoints = multipoints.explode(index_parts=False).reset_index(drop=True)
    summary["multipoint_rows_expanded"] = len(multipoints)

    # Convert polygons and multipolygons into representative points
    polygons = out[out.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    if not polygons.empty:
        polygons["geometry"] = polygons.representative_point()
    summary["polygon_rows_converted"] = len(polygons)

    # Drop unsupported geometry types
    supported_types = {"Point", "MultiPoint", "Polygon", "MultiPolygon"}
    unsupported = out[~out.geometry.geom_type.isin(supported_types)].copy()
    summary["unsupported_rows_dropped"] = len(unsupported)

    combined = pd.concat([points, multipoints, polygons], ignore_index=True)
    combined = gpd.GeoDataFrame(combined, geometry="geometry", crs=gdf.crs)

    return combined, summary


def deduplicate_service_points(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Deduplicate primarily by OSM identity, with a fallback coordinate-based pass.
    """
    out = gdf.copy()

    before = len(out)

    if "osm_element_type" in out.columns and "osm_id" in out.columns:
        out = out.drop_duplicates(subset=["osm_element_type", "osm_id"]).copy()

    # Coordinate fallback for residual exact duplicates
    out["x"] = out.geometry.x
    out["y"] = out.geometry.y

    coord_subset = ["service_type", "x", "y"]
    if "name" in out.columns:
        coord_subset.append("name")

    out = out.drop_duplicates(subset=coord_subset).copy()

    after = len(out)
    print(f"Deduplication removed {before - after} rows.")

    return out


def spatially_assign_services_to_zones(
    service_points: gpd.GeoDataFrame,
    zones_master_2022: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Spatially join each service point to a 2022 Data Zone.
    """
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

    zones = zones_master_2022[zone_cols].copy()

    joined = gpd.sjoin(
        service_points,
        zones,
        how="left",
        predicate="within",
    )

    if "index_right" in joined.columns:
        joined = joined.drop(columns=["index_right"])

    return joined


def keep_useful_columns(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Keep a controlled final set of fields for the processed service-point layer.
    """
    preferred_cols = [
        "osm_element_type",
        "osm_id",
        "amenity",
        "name",
        "brand",
        "operator",
        "addr:street",
        "addr:city",
        "addr:postcode",
        "service_type",
        "source_geometry_type",
        "x",
        "y",
        "dz_code_2022",
        "dz_name_2022",
        "ur6_name",
        "ur8_name",
        "is_rural",
        "is_accessible_rural",
        "is_remote_rural",
        "geometry",
    ]

    out = gdf.copy()
    available_cols = [col for col in preferred_cols if col in out.columns]
    out = out[available_cols].copy()

    return out


def main() -> None:
    print("Starting refined service_points_current build...")

    services_interim_dir = INTERIM_DIR / "services_current"
    services_processed_dir = PROCESSED_DIR / "services_current"
    services_processed_dir.mkdir(parents=True, exist_ok=True)

    zones_path = PROCESSED_DIR / "geography" / "zones_master_2022.gpkg"

    banks_path = services_interim_dir / "raw_banks_osm.gpkg"
    atms_path = services_interim_dir / "raw_atms_osm.gpkg"
    post_path = services_interim_dir / "raw_post_offices_osm.gpkg"

    print(f"Loading zones from: {zones_path}")
    zones_master_2022 = gpd.read_file(zones_path, layer="zones_master_2022")

    print("Loading raw service layers...")
    banks = load_service_layer(banks_path, "raw_banks_osm", "bank")
    atms = load_service_layer(atms_path, "raw_atms_osm", "atm")
    post_offices = load_service_layer(post_path, "raw_post_offices_osm", "post_office")

    print("\n--- Raw layer counts ---")
    print(f"Banks raw rows: {len(banks)}")
    print(f"ATMs raw rows: {len(atms)}")
    print(f"Post offices raw rows: {len(post_offices)}")

    service_points_raw = pd.concat([banks, atms, post_offices], ignore_index=True)
    service_points_raw = gpd.GeoDataFrame(
        service_points_raw,
        geometry="geometry",
        crs=zones_master_2022.crs,
    )

    if service_points_raw.crs != zones_master_2022.crs:
        service_points_raw = service_points_raw.to_crs(zones_master_2022.crs)

    print(f"\nCombined raw service rows: {len(service_points_raw)}")

    print("\nGeometry types before conversion:")
    print(service_points_raw.geometry.geom_type.value_counts(dropna=False))

    service_points, conversion_summary = convert_services_to_points(service_points_raw)

    print("\n--- Geometry conversion summary ---")
    print(f"Original rows: {conversion_summary['original_rows']}")
    print(f"Point rows kept: {conversion_summary['point_rows_kept']}")
    print(f"MultiPoint rows expanded: {conversion_summary['multipoint_rows_expanded']}")
    print(f"Polygon/MultiPolygon rows converted: {conversion_summary['polygon_rows_converted']}")
    print(f"Unsupported rows dropped: {conversion_summary['unsupported_rows_dropped']}")

    print("\nGeometry types after conversion:")
    print(service_points.geometry.geom_type.value_counts(dropna=False))

    print(f"\nRows after point conversion: {len(service_points)}")

    service_points = deduplicate_service_points(service_points)

    print("\nCounts by service_type after deduplication:")
    print(service_points["service_type"].value_counts(dropna=False))

    print("\nSpatially assigning service points to 2022 zones...")
    service_points = spatially_assign_services_to_zones(service_points, zones_master_2022)

    missing_zone_matches = service_points["dz_code_2022"].isna().sum()

    print("\nCounts by service_type after spatial assignment:")
    print(service_points["service_type"].value_counts(dropna=False))

    print(f"\nService points without 2022 zone match: {missing_zone_matches}")

    print("\nMatched rural vs non-rural counts:")
    print(service_points["is_rural"].value_counts(dropna=False))

    # Final coordinate fields for export
    service_points["x"] = service_points.geometry.x
    service_points["y"] = service_points.geometry.y

    service_points = keep_useful_columns(service_points)

    output_gpkg = services_processed_dir / "service_points_current.gpkg"
    output_csv = services_processed_dir / "service_points_current.csv"

    service_points.to_file(output_gpkg, layer="service_points_current", driver="GPKG")
    pd.DataFrame(service_points.drop(columns="geometry")).to_csv(output_csv, index=False)

    print("\nRefined service_points_current build completed successfully.")
    print(f"Output GeoPackage: {output_gpkg}")
    print(f"Output CSV: {output_csv}")


if __name__ == "__main__":
    main()