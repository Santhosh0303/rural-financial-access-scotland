from pathlib import Path
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

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


def clean_text_columns(df: pd.DataFrame, text_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in text_cols:
        if col in out.columns:
            out[col] = out[col].astype("string").str.strip()
    return out


def build_close_date(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["close_year"] = pd.to_numeric(out["close_year"], errors="coerce")
    out["close_month"] = pd.to_numeric(out["close_month"], errors="coerce")

    out["close_month_filled"] = out["close_month"].fillna(1).astype("Int64")
    out["close_date"] = pd.to_datetime(
        dict(
            year=out["close_year"].astype("Int64"),
            month=out["close_month_filled"],
            day=1,
        ),
        errors="coerce",
    )

    out["close_month_missing_flag"] = out["close_month"].isna().astype(int)

    return out


def assign_period_group(year_value) -> str:
    if pd.isna(year_value):
        return "unknown"
    year_value = int(year_value)

    if year_value <= 2019:
        return "pre_covid"
    if year_value == 2020:
        return "covid_transition"
    return "post_covid"


def resolve_geolytix_folder(raw_dir: Path) -> Path:
    """
    Robustly find the Geolytix bank closures folder even if the parent folder
    uses spaces/underscores or slight naming differences.
    """
    preferred_candidates = [
        raw_dir / "bank_closures Geolytix" / "GEOLYTIX - UK Open Banks",
        raw_dir / "bank_closures_Geolytix" / "GEOLYTIX - UK Open Banks",
        raw_dir / "bank_closures" / "GEOLYTIX - UK Open Banks",
    ]

    for candidate in preferred_candidates:
        if candidate.exists():
            return candidate

    # Fallback: search raw directory
    matches = []
    for path in raw_dir.rglob("*"):
        if path.is_dir():
            path_name = path.name.lower()
            parent_name = path.parent.name.lower()

            if (
                "geolytix" in path_name
                and "open banks" in path_name
            ) or (
                "geolytix" in parent_name
                and "open banks" in path_name
            ):
                matches.append(path)

    if matches:
        matches = sorted(matches)
        return matches[0]

    raise FileNotFoundError(
        "Could not find the 'GEOLYTIX - UK Open Banks' folder under data/raw."
    )


def resolve_required_file(folder: Path, filename: str) -> Path:
    """
    Find the required file exactly, with a fallback recursive search.
    """
    direct_path = folder / filename
    if direct_path.exists():
        return direct_path

    matches = list(folder.rglob(filename))
    if matches:
        return matches[0]

    raise FileNotFoundError(f"Missing required file: {filename} under {folder}")


def main() -> None:
    print("Starting bank closure extraction...")

    data_dir = PROCESSED_DIR.parent
    raw_dir = data_dir / "raw"
    interim_dir = data_dir / "interim" / "bank_closures"
    interim_dir.mkdir(parents=True, exist_ok=True)

    geolytix_dir = resolve_geolytix_folder(raw_dir)

    closures_xlsx_path = resolve_required_file(
        geolytix_dir, "bank_closures_scotland_closed_closing_raw.xlsx"
    )
    open_banks_csv_path = resolve_required_file(
        geolytix_dir, "geolytix_uk_open_bank_branches.csv"
    )
    user_guide_path = resolve_required_file(
        geolytix_dir, "GEOLYTIX - UK Banks - User Guide.pdf"
    )
    readme_path = resolve_required_file(
        geolytix_dir, "README.txt"
    )

    print(f"Geolytix source folder: {geolytix_dir}")
    print(f"Closures Excel path: {closures_xlsx_path}")
    print(f"Open banks CSV path: {open_banks_csv_path}")
    print(f"User guide path: {user_guide_path}")
    print(f"README path: {readme_path}")

    print("\nLoading Scotland closures Excel...")
    closures_raw = pd.read_excel(closures_xlsx_path)

    print(f"Rows loaded from closures Excel: {len(closures_raw)}")
    print(f"Columns loaded: {closures_raw.columns.tolist()}")

    missing_expected = [col for col in EXPECTED_COLUMNS if col not in closures_raw.columns]
    if missing_expected:
        raise ValueError(f"Missing expected columns in closures Excel: {missing_expected}")

    closures = closures_raw[EXPECTED_COLUMNS].copy()

    closures = clean_text_columns(
        closures,
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

    closures["status"] = closures["status"].fillna("").str.strip()
    closures["region"] = closures["region"].fillna("").str.strip()

    closures["long_wgs84"] = pd.to_numeric(closures["long_wgs84"], errors="coerce")
    closures["lat_wgs84"] = pd.to_numeric(closures["lat_wgs84"], errors="coerce")
    closures["po_dist"] = pd.to_numeric(closures["po_dist"], errors="coerce")
    closures["open_year"] = pd.to_numeric(closures["open_year"], errors="coerce")

    print("\nFiltering to Scotland region only...")
    closures = closures[closures["region"].str.lower() == "scotland"].copy()

    print("Filtering to closed / closing statuses only...")
    closures = closures[
        closures["status"].str.lower().isin(["closed", "closing", "permanently closed"])
    ].copy()

    print("Dropping rows with missing coordinates...")
    before_coord_drop = len(closures)
    closures = closures.dropna(subset=["long_wgs84", "lat_wgs84"]).copy()
    dropped_missing_coords = before_coord_drop - len(closures)

    closures = build_close_date(closures)
    closures["closure_period_group"] = closures["close_year"].apply(assign_period_group)

    closures["closure_event_flag"] = 1

    closures["branch_name"] = closures["branch_name"].fillna("")
    closures["branch_label"] = closures["branch_name"]
    blank_branch_mask = closures["branch_label"].str.strip() == ""
    closures.loc[blank_branch_mask, "branch_label"] = (
        closures.loc[blank_branch_mask, "brand_full"].fillna("")
        + " "
        + closures.loc[blank_branch_mask, "town"].fillna("")
    ).str.strip()

    closures["closure_year_month"] = closures["close_date"].dt.to_period("M").astype("string")

    closures = closures.sort_values(
        by=["close_year", "close_month_filled", "brand_full", "town", "branch_label"],
        ascending=[True, True, True, True, True],
    ).reset_index(drop=True)

    closures["closure_record_rank"] = closures.index + 1

    print("\nBuilding GeoDataFrame...")
    geometry = [Point(xy) for xy in zip(closures["long_wgs84"], closures["lat_wgs84"])]
    closures_gdf = gpd.GeoDataFrame(closures.copy(), geometry=geometry, crs="EPSG:4326")
    closures_gdf_27700 = closures_gdf.to_crs("EPSG:27700")

    yearly_summary = (
        closures.groupby(["close_year", "closure_period_group"], dropna=False)
        .agg(
            closure_count=("id", "count"),
            unique_brands=("brand_full", "nunique"),
            unique_towns=("town", "nunique"),
            mean_po_dist_m=("po_dist", "mean"),
        )
        .reset_index()
        .sort_values("close_year")
        .reset_index(drop=True)
    )

    brand_year_summary = (
        closures.groupby(["close_year", "brand_full"], dropna=False)
        .size()
        .reset_index(name="closure_count")
        .sort_values(["close_year", "closure_count", "brand_full"], ascending=[True, False, True])
        .reset_index(drop=True)
    )

    branch_type_summary = (
        closures.groupby("branch_type", dropna=False)
        .size()
        .reset_index(name="closure_count")
        .sort_values("closure_count", ascending=False)
        .reset_index(drop=True)
    )

    print("\n--- Bank Closure Extraction Summary ---")
    print(f"Final Scotland closure rows: {len(closures)}")
    print(f"Dropped rows with missing coordinates: {dropped_missing_coords}")

    print("\nStatus counts:")
    print(closures["status"].value_counts(dropna=False))

    print("\nClosure years covered:")
    print(sorted(closures["close_year"].dropna().astype(int).unique().tolist()))

    print("\nClosures by year:")
    print(closures["close_year"].dropna().astype(int).value_counts().sort_index())

    print("\nClosures by branch type:")
    print(closures["branch_type"].value_counts(dropna=False))

    print("\nClosure period counts:")
    print(closures["closure_period_group"].value_counts(dropna=False))

    print("\nTop 10 brands by closure count:")
    print(closures["brand_full"].value_counts(dropna=False).head(10))

    print("\nPreview of cleaned closures:")
    preview_cols = [
        "id",
        "brand_full",
        "branch_label",
        "branch_type",
        "town",
        "postcode",
        "status",
        "close_month",
        "close_year",
        "closure_period_group",
        "po_dist",
    ]
    print(closures[preview_cols].head(10))

    cleaned_csv_path = interim_dir / "bank_closures_scotland_cleaned.csv"
    cleaned_gpkg_path = interim_dir / "bank_closures_scotland_cleaned.gpkg"
    yearly_summary_path = interim_dir / "bank_closures_scotland_yearly_summary.csv"
    brand_year_summary_path = interim_dir / "bank_closures_scotland_brand_year_summary.csv"
    branch_type_summary_path = interim_dir / "bank_closures_scotland_branch_type_summary.csv"

    closures.to_csv(cleaned_csv_path, index=False)
    closures_gdf_27700.to_file(
        cleaned_gpkg_path,
        layer="bank_closures_scotland_cleaned",
        driver="GPKG",
    )
    yearly_summary.to_csv(yearly_summary_path, index=False)
    brand_year_summary.to_csv(brand_year_summary_path, index=False)
    branch_type_summary.to_csv(branch_type_summary_path, index=False)

    print("\nBank closure extraction completed successfully.")
    print(f"Cleaned CSV output: {cleaned_csv_path}")
    print(f"Cleaned GPKG output: {cleaned_gpkg_path}")
    print(f"Yearly summary output: {yearly_summary_path}")
    print(f"Brand-year summary output: {brand_year_summary_path}")
    print(f"Branch-type summary output: {branch_type_summary_path}")


if __name__ == "__main__":
    main()