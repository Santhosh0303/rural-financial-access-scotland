from pathlib import Path
import pandas as pd
import geopandas as gpd
from shapely.geometry import box
import osmnx as ox

from config.paths import PROCESSED_DIR, INTERIM_DIR


GRID_SIZE_M = 50_000  # 50 km tiles
COMPLETE_STATUSES = {"success", "empty", "invalid_tile_skipped"}


def build_study_boundaries(zones_master_2022: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    rural_zones_2022 = zones_master_2022[zones_master_2022["is_rural"] == 1].copy()

    scotland_boundary = zones_master_2022.dissolve().reset_index(drop=True)
    scotland_boundary["boundary_type"] = "scotland_all_zones"

    rural_boundary = rural_zones_2022.dissolve().reset_index(drop=True)
    rural_boundary["boundary_type"] = "scotland_rural_zones_only"

    if not scotland_boundary.is_valid.all():
        print("Fixing invalid geometry in Scotland boundary using buffer(0)...")
        scotland_boundary["geometry"] = scotland_boundary.buffer(0)

    if not rural_boundary.is_valid.all():
        print("Fixing invalid geometry in rural boundary using buffer(0)...")
        rural_boundary["geometry"] = rural_boundary.buffer(0)

    scotland_boundary["area_sqkm"] = scotland_boundary.geometry.area / 1_000_000
    rural_boundary["area_sqkm"] = rural_boundary.geometry.area / 1_000_000

    return scotland_boundary, rural_boundary


def build_extraction_tiles(boundary_gdf: gpd.GeoDataFrame, grid_size_m: int = GRID_SIZE_M) -> gpd.GeoDataFrame:
    """
    Build square tiles over the Scotland study boundary and clip them to the boundary.
    """
    boundary = boundary_gdf.copy()

    minx, miny, maxx, maxy = boundary.total_bounds

    x_coords = list(range(int(minx), int(maxx) + grid_size_m, grid_size_m))
    y_coords = list(range(int(miny), int(maxy) + grid_size_m, grid_size_m))

    cells = []
    tile_id = 1
    for x in x_coords[:-1]:
        for y in y_coords[:-1]:
            geom = box(x, y, x + grid_size_m, y + grid_size_m)
            cells.append({"tile_id": tile_id, "geometry": geom})
            tile_id += 1

    grid = gpd.GeoDataFrame(cells, geometry="geometry", crs=boundary.crs)

    # Keep tiles intersecting the Scotland study boundary
    grid = grid[grid.intersects(boundary.geometry.iloc[0])].copy()

    # Clip tile geometries to the boundary
    grid = gpd.overlay(grid, boundary[["geometry"]], how="intersection", keep_geom_type=False)

    # Keep only polygon-based geometries
    valid_geom_types = ["Polygon", "MultiPolygon"]
    grid = grid[grid.geometry.geom_type.isin(valid_geom_types)].copy()

    invalid_mask = ~grid.is_valid
    if invalid_mask.any():
        print(f"Fixing {invalid_mask.sum()} invalid extraction tiles using buffer(0)...")
        grid.loc[invalid_mask, "geometry"] = grid.loc[invalid_mask, "geometry"].buffer(0)

    grid = grid[grid.geometry.area > 0].copy()
    grid["tile_area_sqkm"] = grid.geometry.area / 1_000_000

    grid = grid.reset_index(drop=True)
    grid["tile_seq"] = grid.index + 1

    return grid


def reset_osm_index(features: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = features.reset_index()

    rename_map = {}
    if "element" in gdf.columns:
        rename_map["element"] = "osm_element_type"
    if "element_type" in gdf.columns:
        rename_map["element_type"] = "osm_element_type"

    if "id" in gdf.columns:
        rename_map["id"] = "osm_id"
    if "osmid" in gdf.columns:
        rename_map["osmid"] = "osm_id"

    gdf = gdf.rename(columns=rename_map)
    return gdf


def standardise_service_type(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = gdf.copy()
    out["service_type"] = None

    if "amenity" in out.columns:
        out.loc[out["amenity"] == "bank", "service_type"] = "bank"
        out.loc[out["amenity"] == "atm", "service_type"] = "atm"
        out.loc[out["amenity"] == "post_office", "service_type"] = "post_office"

    return out


def keep_useful_columns(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
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
        "tile_id",
        "tile_seq",
        "geometry",
    ]

    out = gdf.copy()
    for col in preferred_cols:
        if col not in out.columns and col != "geometry":
            out[col] = None

    return out[preferred_cols].copy()


def deduplicate_osm_features(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = gdf.copy()

    if "osm_element_type" in out.columns and "osm_id" in out.columns:
        out = out.drop_duplicates(subset=["osm_element_type", "osm_id"]).copy()
    else:
        out = out.drop_duplicates().copy()

    return out


def load_existing_tile_log(log_path: Path) -> pd.DataFrame:
    if log_path.exists():
        log_df = pd.read_csv(log_path)
        if "tile_id" in log_df.columns:
            log_df = log_df.sort_values("tile_id").drop_duplicates(subset=["tile_id"], keep="last")
        return log_df
    return pd.DataFrame(columns=["tile_id", "tile_seq", "status", "rows", "error_message"])


def save_tile_log(log_df: pd.DataFrame, log_path: Path) -> None:
    log_df = log_df.sort_values("tile_id").drop_duplicates(subset=["tile_id"], keep="last")
    log_df.to_csv(log_path, index=False)


def extract_combined_osm_features_by_tiles_with_resume(
    tiles_gdf: gpd.GeoDataFrame,
    checkpoint_dir: Path,
    log_path: Path,
) -> pd.DataFrame:
    """
    Query each tile once for bank, atm, and post_office together, saving progress tile by tile.
    """
    ox.settings.use_cache = True
    ox.settings.requests_timeout = 300

    tags = {"amenity": ["bank", "atm", "post_office"]}
    tiles_wgs84 = tiles_gdf.to_crs(epsg=4326)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    tile_log = load_existing_tile_log(log_path)
    completed_tile_ids = set(
        tile_log.loc[tile_log["status"].isin(COMPLETE_STATUSES), "tile_id"].astype(int).tolist()
    )

    total_tiles = len(tiles_wgs84)

    for _, row in tiles_wgs84.iterrows():
        tile_id = int(row["tile_id"])
        tile_seq = int(row["tile_seq"])
        polygon = row.geometry

        if tile_id in completed_tile_ids:
            print(f"  Skipping tile {tile_seq}/{total_tiles} (tile_id={tile_id}) - already completed.")
            continue

        print(f"  Querying combined services | tile {tile_seq}/{total_tiles} (tile_id={tile_id}) ...")

        if not polygon.is_valid:
            polygon = polygon.buffer(0)

        if not polygon.is_valid or polygon.is_empty:
            row_log = pd.DataFrame(
                [{
                    "tile_id": tile_id,
                    "tile_seq": tile_seq,
                    "status": "invalid_tile_skipped",
                    "rows": 0,
                    "error_message": "Tile geometry invalid after repair",
                }]
            )
            tile_log = pd.concat([tile_log, row_log], ignore_index=True)
            save_tile_log(tile_log, log_path)
            print(f"    Tile {tile_id} skipped: invalid geometry after repair.")
            continue

        try:
            features = ox.features_from_polygon(polygon, tags=tags)

            if features.empty:
                row_log = pd.DataFrame(
                    [{
                        "tile_id": tile_id,
                        "tile_seq": tile_seq,
                        "status": "empty",
                        "rows": 0,
                        "error_message": None,
                    }]
                )
                tile_log = pd.concat([tile_log, row_log], ignore_index=True)
                save_tile_log(tile_log, log_path)
                print(f"    Tile {tile_id} empty.")
                continue

            features = reset_osm_index(features)
            features = gpd.GeoDataFrame(features, geometry="geometry", crs="EPSG:4326")
            features["tile_id"] = tile_id
            features["tile_seq"] = tile_seq

            tile_output_path = checkpoint_dir / f"tile_{tile_id:03d}_services.gpkg"
            features.to_file(tile_output_path, layer="tile_services", driver="GPKG")

            row_log = pd.DataFrame(
                [{
                    "tile_id": tile_id,
                    "tile_seq": tile_seq,
                    "status": "success",
                    "rows": len(features),
                    "error_message": None,
                }]
            )
            tile_log = pd.concat([tile_log, row_log], ignore_index=True)
            save_tile_log(tile_log, log_path)
            print(f"    Tile {tile_id} saved with {len(features)} raw features.")

        except Exception as exc:
            msg = str(exc)

            if "No matching features" in msg:
                row_log = pd.DataFrame(
                    [{
                        "tile_id": tile_id,
                        "tile_seq": tile_seq,
                        "status": "empty",
                        "rows": 0,
                        "error_message": None,
                    }]
                )
                tile_log = pd.concat([tile_log, row_log], ignore_index=True)
                save_tile_log(tile_log, log_path)
                print(f"    Tile {tile_id} empty.")
            else:
                row_log = pd.DataFrame(
                    [{
                        "tile_id": tile_id,
                        "tile_seq": tile_seq,
                        "status": "failed",
                        "rows": 0,
                        "error_message": msg,
                    }]
                )
                tile_log = pd.concat([tile_log, row_log], ignore_index=True)
                save_tile_log(tile_log, log_path)
                print(f"    Tile {tile_id} failed: {msg}")

    tile_log = load_existing_tile_log(log_path)
    return tile_log


def combine_checkpoint_tiles(checkpoint_dir: Path) -> gpd.GeoDataFrame:
    tile_files = sorted(checkpoint_dir.glob("tile_*_services.gpkg"))

    if not tile_files:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    parts = []
    for file_path in tile_files:
        gdf = gpd.read_file(file_path, layer="tile_services")
        parts.append(gdf)

    combined = pd.concat(parts, ignore_index=True)
    combined = gpd.GeoDataFrame(combined, geometry="geometry", crs="EPSG:4326")
    return combined


def split_service_types(combined_gdf: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    out = standardise_service_type(combined_gdf)
    out = keep_useful_columns(out)
    out = deduplicate_osm_features(out)

    banks = out[out["service_type"] == "bank"].copy()
    atms = out[out["service_type"] == "atm"].copy()
    post_offices = out[out["service_type"] == "post_office"].copy()

    return banks, atms, post_offices


def main() -> None:
    print("Starting current-service extraction...")

    geography_path = PROCESSED_DIR / "geography" / "zones_master_2022.gpkg"
    services_dir = INTERIM_DIR / "services_current"
    services_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = services_dir / "checkpoint_tiles"
    extraction_log_path = services_dir / "osm_extraction_tile_log.csv"

    print(f"Loading 2022 master geography from: {geography_path}")
    zones_master_2022 = gpd.read_file(geography_path, layer="zones_master_2022")

    print(f"Rows loaded: {len(zones_master_2022)}")
    print(f"CRS: {zones_master_2022.crs}")
    print(f"Total 2022 zones: {len(zones_master_2022)}")
    print(f"Rural 2022 zones: {(zones_master_2022['is_rural'] == 1).sum()}")

    print("\nBuilding study boundaries...")
    scotland_boundary, rural_boundary = build_study_boundaries(zones_master_2022)

    scotland_boundary_path = services_dir / "scotland_study_boundary.gpkg"
    rural_boundary_path = services_dir / "rural_study_boundary.gpkg"

    scotland_boundary.to_file(
        scotland_boundary_path,
        layer="scotland_study_boundary",
        driver="GPKG",
    )

    rural_boundary.to_file(
        rural_boundary_path,
        layer="rural_study_boundary",
        driver="GPKG",
    )

    print("\nBuilding extraction tiles over the full Scotland boundary...")
    extraction_tiles = build_extraction_tiles(scotland_boundary, grid_size_m=GRID_SIZE_M)

    tiles_path = services_dir / "extraction_tiles_scotland.gpkg"
    extraction_tiles.to_file(
        tiles_path,
        layer="extraction_tiles_scotland",
        driver="GPKG",
    )

    print("\n--- Study Boundary and Tile Summary ---")
    print(f"Scotland boundary rows: {len(scotland_boundary)}")
    print(f"Rural boundary rows: {len(rural_boundary)}")
    print(f"Scotland boundary area (sq km): {scotland_boundary['area_sqkm'].iloc[0]:.4f}")
    print(f"Rural boundary area (sq km): {rural_boundary['area_sqkm'].iloc[0]:.4f}")
    print(f"Extraction tile rows: {len(extraction_tiles)}")
    print(f"Mean extraction tile area (sq km): {extraction_tiles['tile_area_sqkm'].mean():.4f}")

    print("\nRunning combined tile extraction with checkpoint/resume...")
    extraction_log = extract_combined_osm_features_by_tiles_with_resume(
        tiles_gdf=extraction_tiles,
        checkpoint_dir=checkpoint_dir,
        log_path=extraction_log_path,
    )

    print("\nCombining completed tile outputs...")
    combined_raw = combine_checkpoint_tiles(checkpoint_dir)

    if not combined_raw.empty:
        combined_raw = combined_raw.to_crs(zones_master_2022.crs)

    banks_raw, atms_raw, post_raw = split_service_types(combined_raw)

    raw_combined_path = services_dir / "raw_services_combined_osm.gpkg"
    raw_banks_path = services_dir / "raw_banks_osm.gpkg"
    raw_atms_path = services_dir / "raw_atms_osm.gpkg"
    raw_post_path = services_dir / "raw_post_offices_osm.gpkg"

    if not combined_raw.empty:
        combined_raw.to_file(raw_combined_path, layer="raw_services_combined_osm", driver="GPKG")
    if not banks_raw.empty:
        banks_raw.to_file(raw_banks_path, layer="raw_banks_osm", driver="GPKG")
    if not atms_raw.empty:
        atms_raw.to_file(raw_atms_path, layer="raw_atms_osm", driver="GPKG")
    if not post_raw.empty:
        post_raw.to_file(raw_post_path, layer="raw_post_offices_osm", driver="GPKG")

    print("\n--- Raw OSM Extraction Counts After Checkpoint/Resume Combination ---")
    print(f"Combined features extracted: {len(combined_raw)}")
    print(f"Banks extracted: {len(banks_raw)}")
    print(f"ATMs extracted: {len(atms_raw)}")
    print(f"Post offices extracted: {len(post_raw)}")

    print("\nTile extraction status summary:")
    if not extraction_log.empty:
        print(extraction_log.groupby("status").size())
    else:
        print("No tile log rows found.")

    print("\nCurrent-service extraction completed successfully.")
    print(f"Scotland study boundary saved to: {scotland_boundary_path}")
    print(f"Rural study boundary saved to: {rural_boundary_path}")
    print(f"Extraction tiles saved to: {tiles_path}")
    print(f"Checkpoint tile folder: {checkpoint_dir}")
    print(f"Combined raw services output: {raw_combined_path}")
    print(f"Raw banks output: {raw_banks_path}")
    print(f"Raw ATMs output: {raw_atms_path}")
    print(f"Raw post offices output: {raw_post_path}")
    print(f"Tile extraction log output: {extraction_log_path}")


if __name__ == "__main__":
    main()