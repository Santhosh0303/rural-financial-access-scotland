import pandas as pd
import geopandas as gpd

from config.paths import PROCESSED_DIR


PANEL_START_YEAR = 2015
PANEL_END_YEAR = 2023


def assign_period_group(year_value) -> str:
    if pd.isna(year_value):
        return "unknown"
    year_value = int(year_value)

    if year_value <= 2019:
        return "pre_covid"
    if year_value == 2020:
        return "covid_transition"
    return "post_covid"


def main() -> None:
    print("Starting bank closure panel build...")

    interim_dir = PROCESSED_DIR.parent / "interim" / "bank_closures"
    processed_dir = PROCESSED_DIR / "bank_closures"
    processed_dir.mkdir(parents=True, exist_ok=True)

    closures_path = interim_dir / "bank_closures_scotland_cleaned.gpkg"
    zones_path = PROCESSED_DIR / "geography" / "zones_master_2022.gpkg"
    context_path = PROCESSED_DIR / "context" / "zone_year_context_2022.csv"

    print(f"Loading cleaned closures from: {closures_path}")
    closures = gpd.read_file(closures_path, layer="bank_closures_scotland_cleaned")

    print(f"Loading 2022 zones from: {zones_path}")
    zones = gpd.read_file(zones_path, layer="zones_master_2022")

    print(f"Loading harmonised context from: {context_path}")
    context = pd.read_csv(context_path)

    print(f"Closure rows loaded: {len(closures)}")
    print(f"Zone rows loaded: {len(zones)}")
    print(f"Context rows loaded: {len(context)}")

    if closures.crs != zones.crs:
        closures = closures.to_crs(zones.crs)

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
    zone_cols = [col for col in zone_cols if col in zones.columns]
    zones_subset = zones[zone_cols].copy()

    print("\nSpatially assigning closures to 2022 Data Zones...")
    closures_assigned = gpd.sjoin(
        closures,
        zones_subset,
        how="left",
        predicate="within",
    ).drop(columns=["index_right"], errors="ignore")

    unmatched = int(closures_assigned["dz_code_2022"].isna().sum())

    print("\nAssigned closure summary:")
    print(f"Matched closures: {len(closures_assigned) - unmatched}")
    print(f"Unmatched closures: {unmatched}")

    print("\nClosure counts by rural flag:")
    if "is_rural" in closures_assigned.columns:
        print(closures_assigned["is_rural"].value_counts(dropna=False))

    print("\nClosure counts by UR6:")
    if "ur6_name" in closures_assigned.columns:
        print(closures_assigned["ur6_name"].value_counts(dropna=False))

    # Keep only panel-aligned years for zone-year analysis
    aligned_closures = closures_assigned[
        closures_assigned["close_year"].between(PANEL_START_YEAR, PANEL_END_YEAR, inclusive="both")
    ].copy()

    print(f"\nAligned closure rows ({PANEL_START_YEAR}-{PANEL_END_YEAR}): {len(aligned_closures)}")

    yearly_zone_counts = (
        aligned_closures.groupby(["dz_code_2022", "close_year"], dropna=False)
        .agg(
            closures_in_year=("id", "count"),
            unique_closing_brands=("brand_full", "nunique"),
            unique_closing_branch_types=("branch_type", "nunique"),
        )
        .reset_index()
        .rename(columns={"close_year": "year"})
    )

    years = pd.DataFrame({"year": list(range(PANEL_START_YEAR, PANEL_END_YEAR + 1))})

    zone_lookup_cols = [
        "dz_code_2022",
        "dz_name_2022",
        "ur6_name",
        "ur8_name",
        "is_rural",
        "is_accessible_rural",
        "is_remote_rural",
    ]
    zone_lookup_cols = [col for col in zone_lookup_cols if col in zones_subset.columns]
    zone_lookup = zones_subset[zone_lookup_cols].drop_duplicates(subset=["dz_code_2022"]).copy()

    base_panel = zone_lookup.merge(years, how="cross")

    panel = base_panel.merge(
        yearly_zone_counts,
        on=["dz_code_2022", "year"],
        how="left",
        validate="1:1",
    )

    fill_zero_cols = [
        "closures_in_year",
        "unique_closing_brands",
        "unique_closing_branch_types",
    ]
    for col in fill_zero_cols:
        if col in panel.columns:
            panel[col] = panel[col].fillna(0).astype(int)

    panel["closure_event_flag"] = (panel["closures_in_year"] > 0).astype(int)

    panel = panel.sort_values(["dz_code_2022", "year"]).reset_index(drop=True)

    panel["cumulative_closures_to_year"] = (
        panel.groupby("dz_code_2022")["closures_in_year"].cumsum()
    )

    panel["cumulative_closure_flag"] = (panel["cumulative_closures_to_year"] > 0).astype(int)

    first_closure_lookup = (
        aligned_closures.groupby("dz_code_2022", dropna=False)["close_year"]
        .min()
        .reset_index()
        .rename(columns={"close_year": "first_closure_year"})
    )

    panel = panel.merge(
        first_closure_lookup,
        on="dz_code_2022",
        how="left",
        validate="m:1",
    )

    panel["closure_period_group"] = panel["year"].apply(assign_period_group)

    # Add selected context variables for aligned years
    context_keep_cols = [
        "dz_code_2022",
        "year",
        "population_total",
        "older_population_65_plus",
        "older_population_share",
        "active_simd_rank_overall",
        "active_access_domain_rank",
    ]
    context_keep_cols = [col for col in context_keep_cols if col in context.columns]

    context_aligned = context[context_keep_cols].copy()
    context_aligned = context_aligned[
        context_aligned["year"].between(PANEL_START_YEAR, PANEL_END_YEAR, inclusive="both")
    ].copy()

    panel = panel.merge(
        context_aligned,
        on=["dz_code_2022", "year"],
        how="left",
        validate="1:1",
    )

    yearly_summary = (
        panel.groupby(["year", "closure_period_group"], dropna=False)
        .agg(
            zones_with_closure_event=("closure_event_flag", "sum"),
            total_closures_in_year=("closures_in_year", "sum"),
            cumulative_closures_total=("cumulative_closures_to_year", "sum"),
        )
        .reset_index()
        .sort_values("year")
        .reset_index(drop=True)
    )

    rural_yearly_summary = (
        panel[panel["is_rural"] == 1]
        .groupby(["year", "ur6_name"], dropna=False)
        .agg(
            rural_zone_count=("dz_code_2022", "count"),
            zones_with_closure_event=("closure_event_flag", "sum"),
            total_closures_in_year=("closures_in_year", "sum"),
            cumulative_closures_total=("cumulative_closures_to_year", "sum"),
        )
        .reset_index()
        .sort_values(["year", "ur6_name"])
        .reset_index(drop=True)
    )

    zone_summary = (
        panel.groupby(
            [
                "dz_code_2022",
                "dz_name_2022",
                "ur6_name",
                "ur8_name",
                "is_rural",
                "is_accessible_rural",
                "is_remote_rural",
            ],
            dropna=False,
        )
        .agg(
            total_closures_2015_2023=("closures_in_year", "sum"),
            years_with_closure_event=("closure_event_flag", "sum"),
            first_closure_year=("first_closure_year", "min"),
            final_cumulative_closures_2023=("cumulative_closures_to_year", "max"),
        )
        .reset_index()
        .sort_values(
            ["total_closures_2015_2023", "years_with_closure_event"],
            ascending=False,
        )
        .reset_index(drop=True)
    )

    print("\n--- Bank Closure Panel Summary ---")
    print(f"Zone-year panel rows: {len(panel)}")
    print(f"Unique 2022 zones in panel: {panel['dz_code_2022'].nunique()}")
    print(f"Years in panel: {sorted(panel['year'].unique().tolist())}")

    print("\nTotal closures in aligned panel by year:")
    print(
        panel.groupby("year")["closures_in_year"].sum()
    )

    print("\nRural vs non-rural total closures (2015-2023):")
    print(
        panel.groupby("is_rural")["closures_in_year"].sum()
    )

    print("\nTop 10 zones by total closures (2015-2023):")
    print(
        zone_summary[
            [
                "dz_code_2022",
                "dz_name_2022",
                "ur6_name",
                "total_closures_2015_2023",
                "years_with_closure_event",
                "first_closure_year",
            ]
        ].head(10)
    )

    assigned_csv_path = processed_dir / "bank_closures_scotland_assigned_2022.csv"
    assigned_gpkg_path = processed_dir / "bank_closures_scotland_assigned_2022.gpkg"
    panel_csv_path = processed_dir / "zone_year_bank_closure_panel_2022.csv"
    yearly_summary_path = processed_dir / "bank_closure_yearly_summary_2022.csv"
    rural_yearly_summary_path = processed_dir / "bank_closure_rural_yearly_summary_2022.csv"
    zone_summary_path = processed_dir / "bank_closure_zone_summary_2022.csv"

    pd.DataFrame(closures_assigned.drop(columns="geometry")).to_csv(assigned_csv_path, index=False)
    closures_assigned.to_file(
        assigned_gpkg_path,
        layer="bank_closures_scotland_assigned_2022",
        driver="GPKG",
    )
    panel.to_csv(panel_csv_path, index=False)
    yearly_summary.to_csv(yearly_summary_path, index=False)
    rural_yearly_summary.to_csv(rural_yearly_summary_path, index=False)
    zone_summary.to_csv(zone_summary_path, index=False)

    print("\nBank closure panel build completed successfully.")
    print(f"Assigned closures CSV: {assigned_csv_path}")
    print(f"Assigned closures GPKG: {assigned_gpkg_path}")
    print(f"Zone-year panel CSV: {panel_csv_path}")
    print(f"Yearly summary CSV: {yearly_summary_path}")
    print(f"Rural yearly summary CSV: {rural_yearly_summary_path}")
    print(f"Zone summary CSV: {zone_summary_path}")


if __name__ == "__main__":
    main()