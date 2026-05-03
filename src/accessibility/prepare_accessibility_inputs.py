from pathlib import Path
import pandas as pd
import geopandas as gpd

from config.paths import PROCESSED_DIR


def build_zone_origin_points(zones_master_2022: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Convert 2022 Data Zone polygons into representative origin points.
    representative_point() is used instead of centroid so the point is guaranteed
    to lie inside the zone polygon.
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

    missing = [col for col in zone_cols if col not in zones_master_2022.columns]
    if missing:
        raise ValueError(f"Missing required zone columns: {missing}")

    origins = zones_master_2022[zone_cols].copy()
    origins["geometry"] = origins.representative_point()
    origins["origin_x"] = origins.geometry.x
    origins["origin_y"] = origins.geometry.y

    if origins["dz_code_2022"].duplicated().any():
        raise ValueError("Duplicate dz_code_2022 values found in origin layer.")

    return origins


def prepare_service_destinations(service_points_current: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Keep the clean point-based service inventory needed for accessibility work.
    """
    required_cols = [
        "osm_element_type",
        "osm_id",
        "service_type",
        "dz_code_2022",
        "dz_name_2022",
        "ur6_name",
        "ur8_name",
        "is_rural",
        "is_accessible_rural",
        "is_remote_rural",
        "geometry",
    ]

    missing = [col for col in required_cols if col not in service_points_current.columns]
    if missing:
        raise ValueError(f"Missing required service columns: {missing}")

    destinations = service_points_current.copy()

    valid_service_types = {"bank", "atm", "post_office"}
    destinations = destinations[destinations["service_type"].isin(valid_service_types)].copy()

    if not destinations.geometry.geom_type.isin(["Point"]).all():
        raise ValueError("Destination layer contains non-point geometries.")

    destinations["dest_x"] = destinations.geometry.x
    destinations["dest_y"] = destinations.geometry.y

    return destinations


def main() -> None:
    print("Starting accessibility input preparation...")

    geography_path = PROCESSED_DIR / "geography" / "zones_master_2022.gpkg"
    services_path = PROCESSED_DIR / "services_current" / "service_points_current.gpkg"
    accessibility_dir = PROCESSED_DIR / "accessibility"
    accessibility_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading geography from: {geography_path}")
    zones_master_2022 = gpd.read_file(geography_path, layer="zones_master_2022")

    print(f"Loading services from: {services_path}")
    service_points_current = gpd.read_file(services_path, layer="service_points_current")

    print(f"Zones rows loaded: {len(zones_master_2022)}")
    print(f"Service rows loaded: {len(service_points_current)}")

    if zones_master_2022.crs != service_points_current.crs:
        service_points_current = service_points_current.to_crs(zones_master_2022.crs)

    print("\nBuilding zone origin points...")
    zone_origins_2022 = build_zone_origin_points(zones_master_2022)

    print("Preparing destination service points...")
    service_destinations_current = prepare_service_destinations(service_points_current)

    origins_output_gpkg = accessibility_dir / "zone_origins_2022.gpkg"
    origins_output_csv = accessibility_dir / "zone_origins_2022.csv"

    destinations_output_gpkg = accessibility_dir / "service_destinations_current.gpkg"
    destinations_output_csv = accessibility_dir / "service_destinations_current.csv"

    zone_origins_2022.to_file(
        origins_output_gpkg,
        layer="zone_origins_2022",
        driver="GPKG",
    )
    pd.DataFrame(zone_origins_2022.drop(columns="geometry")).to_csv(origins_output_csv, index=False)

    service_destinations_current.to_file(
        destinations_output_gpkg,
        layer="service_destinations_current",
        driver="GPKG",
    )
    pd.DataFrame(service_destinations_current.drop(columns="geometry")).to_csv(
        destinations_output_csv,
        index=False,
    )

    print("\n--- Accessibility Input Summary ---")
    print(f"Zone origins rows: {len(zone_origins_2022)}")
    print(f"Unique 2022 zones in origins: {zone_origins_2022['dz_code_2022'].nunique()}")

    print("\nOrigin rural vs non-rural counts:")
    print(zone_origins_2022["is_rural"].value_counts(dropna=False))

    print("\nDestination counts by service_type:")
    print(service_destinations_current["service_type"].value_counts(dropna=False))

    print("\nDestination rural vs non-rural counts:")
    print(service_destinations_current["is_rural"].value_counts(dropna=False))

    print("\nAccessibility input preparation completed successfully.")
    print(f"Zone origins GPKG: {origins_output_gpkg}")
    print(f"Zone origins CSV: {origins_output_csv}")
    print(f"Service destinations GPKG: {destinations_output_gpkg}")
    print(f"Service destinations CSV: {destinations_output_csv}")


if __name__ == "__main__":
    main()