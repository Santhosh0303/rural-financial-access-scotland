from pathlib import Path
import re
import numpy as np
import pandas as pd
import geopandas as gpd

from config.paths import (
    POPULATION_DIR,
    SIMD_2016_FILE,
    SIMD_2020_FILE,
    INTERIM_DIR,
    PROCESSED_DIR,
)


def extract_year_from_name(file_path: Path) -> int | None:
    match = re.search(r"(20\d{2})", file_path.stem)
    if match:
        return int(match.group(1))
    return None


def choose_population_sheet(sheet_names: list[str], year: int) -> str:
    preferred_names = [str(year), "Persons", "persons"]

    for preferred in preferred_names:
        if preferred in sheet_names:
            return preferred

    for sheet in sheet_names:
        lowered = sheet.strip().lower()
        if lowered not in ["cover sheet", "table of contents"]:
            return sheet

    raise ValueError(f"Could not identify a population data sheet from: {sheet_names}")


def detect_header_row(file_path: Path, sheet_name: str, max_rows: int = 10) -> int:
    preview = pd.read_excel(file_path, sheet_name=sheet_name, header=None, nrows=max_rows)

    for idx in range(len(preview)):
        row_values = preview.iloc[idx].astype(str).str.strip().str.lower().tolist()
        if "data zone code" in row_values:
            return idx

    raise ValueError(
        f"Could not detect header row in file {file_path.name}, sheet {sheet_name}"
    )


def get_older_age_columns(columns: list[str]) -> list[str]:
    older_cols = []
    for col in columns:
        col_str = str(col).strip()
        if col_str.startswith("Age "):
            if col_str == "Age 90 and over":
                older_cols.append(col_str)
            else:
                match = re.match(r"Age (\d+)$", col_str)
                if match and int(match.group(1)) >= 65:
                    older_cols.append(col_str)
    return older_cols


def clean_population_file(file_path: Path, year: int) -> pd.DataFrame:
    xl = pd.ExcelFile(file_path)
    sheet_name = choose_population_sheet(xl.sheet_names, year)
    header_row = detect_header_row(file_path, sheet_name)

    df = pd.read_excel(file_path, sheet_name=sheet_name, header=header_row)
    df.columns = [str(col).strip() for col in df.columns]

    required_base_cols = [
        "Data zone code",
        "Data zone name",
        "Council area code",
        "Council area name",
        "Total population",
    ]

    if "Sex" in df.columns:
        df = df[df["Sex"].astype(str).str.strip().str.lower() == "persons"].copy()

    older_age_cols = get_older_age_columns(df.columns.tolist())

    missing_base = [col for col in required_base_cols if col not in df.columns]
    if missing_base:
        raise ValueError(
            f"Missing required base columns in {file_path.name}: {missing_base}"
        )

    if not older_age_cols:
        raise ValueError(f"No 65+ age columns found in {file_path.name}")

    keep_cols = required_base_cols + older_age_cols
    df = df[keep_cols].copy()

    df = df.rename(
        columns={
            "Data zone code": "dz_code_2011",
            "Data zone name": "dz_name_2011",
            "Council area code": "council_area_code",
            "Council area name": "council_area_name",
            "Total population": "population_total",
        }
    )

    df["population_total"] = pd.to_numeric(df["population_total"], errors="coerce")

    for col in older_age_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["older_population_65_plus"] = df[older_age_cols].sum(axis=1)
    df["older_population_share"] = df["older_population_65_plus"] / df["population_total"]
    df["year"] = year

    final_cols = [
        "dz_code_2011",
        "dz_name_2011",
        "council_area_code",
        "council_area_name",
        "year",
        "population_total",
        "older_population_65_plus",
        "older_population_share",
    ]

    return df[final_cols].copy()


def choose_simd_sheet(sheet_names: list[str], preferred_name: str) -> str:
    for sheet in sheet_names:
        if sheet.strip().lower() == preferred_name.strip().lower():
            return sheet
    raise ValueError(f"Could not find SIMD sheet '{preferred_name}' in {sheet_names}")


def clean_simd_2016(file_path: Path) -> pd.DataFrame:
    xl = pd.ExcelFile(file_path)
    sheet_name = choose_simd_sheet(xl.sheet_names, "SIMD16 ranks")

    df = pd.read_excel(file_path, sheet_name=sheet_name)
    df.columns = [str(col).strip() for col in df.columns]

    required_cols = [
        "Data_Zone",
        "Intermediate_Zone",
        "Council_area",
        "Overall_SIMD16_rank",
        "Income_domain_2016_rank",
        "Employment_domain_2016_rank",
        "Health_domain_2016_rank",
        "Education_domain_2016_rank",
        "Housing_domain_2016_rank",
        "Access_domain_2016_rank",
        "Crime_domain_2016_rank",
    ]

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required SIMD 2016 columns: {missing}")

    df = df[required_cols].copy()

    df = df.rename(
        columns={
            "Data_Zone": "dz_code_2011",
            "Intermediate_Zone": "intermediate_zone",
            "Council_area": "council_area_name",
            "Overall_SIMD16_rank": "simd_rank_overall",
            "Income_domain_2016_rank": "income_domain_rank",
            "Employment_domain_2016_rank": "employment_domain_rank",
            "Health_domain_2016_rank": "health_domain_rank",
            "Education_domain_2016_rank": "education_domain_rank",
            "Housing_domain_2016_rank": "housing_domain_rank",
            "Access_domain_2016_rank": "access_domain_rank",
            "Crime_domain_2016_rank": "crime_domain_rank",
        }
    )

    df["simd_year"] = 2016

    final_cols = [
        "dz_code_2011",
        "intermediate_zone",
        "council_area_name",
        "simd_year",
        "simd_rank_overall",
        "income_domain_rank",
        "employment_domain_rank",
        "health_domain_rank",
        "education_domain_rank",
        "housing_domain_rank",
        "access_domain_rank",
        "crime_domain_rank",
    ]

    return df[final_cols].copy()


def clean_simd_2020(file_path: Path) -> pd.DataFrame:
    xl = pd.ExcelFile(file_path)
    sheet_name = choose_simd_sheet(xl.sheet_names, "SIMD 2020v2 ranks")

    df = pd.read_excel(file_path, sheet_name=sheet_name)
    df.columns = [str(col).strip() for col in df.columns]

    required_cols = [
        "Data_Zone",
        "Intermediate_Zone",
        "Council_area",
        "SIMD2020v2_Rank",
        "SIMD2020v2_Income_Domain_Rank",
        "SIMD2020_Employment_Domain_Rank",
        "SIMD2020_Health_Domain_Rank",
        "SIMD2020_Education_Domain_Rank",
        "SIMD2020_Housing_Domain_Rank",
        "SIMD2020_Access_Domain_Rank",
        "SIMD2020_Crime_Domain_Rank",
    ]

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required SIMD 2020 columns: {missing}")

    df = df[required_cols].copy()

    df = df.rename(
        columns={
            "Data_Zone": "dz_code_2011",
            "Intermediate_Zone": "intermediate_zone",
            "Council_area": "council_area_name",
            "SIMD2020v2_Rank": "simd_rank_overall",
            "SIMD2020v2_Income_Domain_Rank": "income_domain_rank",
            "SIMD2020_Employment_Domain_Rank": "employment_domain_rank",
            "SIMD2020_Health_Domain_Rank": "health_domain_rank",
            "SIMD2020_Education_Domain_Rank": "education_domain_rank",
            "SIMD2020_Housing_Domain_Rank": "housing_domain_rank",
            "SIMD2020_Access_Domain_Rank": "access_domain_rank",
            "SIMD2020_Crime_Domain_Rank": "crime_domain_rank",
        }
    )

    df["simd_year"] = 2020

    final_cols = [
        "dz_code_2011",
        "intermediate_zone",
        "council_area_name",
        "simd_year",
        "simd_rank_overall",
        "income_domain_rank",
        "employment_domain_rank",
        "health_domain_rank",
        "education_domain_rank",
        "housing_domain_rank",
        "access_domain_rank",
        "crime_domain_rank",
    ]

    return df[final_cols].copy()


def assign_period_group(year: int) -> str:
    if year <= 2019:
        return "pre_covid"
    if year == 2020:
        return "covid_transition"
    return "post_covid"


def build_simd_wide(simd_panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    simd_value_cols = [
        "simd_rank_overall",
        "income_domain_rank",
        "employment_domain_rank",
        "health_domain_rank",
        "education_domain_rank",
        "housing_domain_rank",
        "access_domain_rank",
        "crime_domain_rank",
    ]

    simd_2016 = simd_panel[simd_panel["simd_year"] == 2016][["dz_code_2011"] + simd_value_cols].copy()
    simd_2020 = simd_panel[simd_panel["simd_year"] == 2020][["dz_code_2011"] + simd_value_cols].copy()

    simd_2016 = simd_2016.rename(columns={col: f"{col}_2016" for col in simd_value_cols})
    simd_2020 = simd_2020.rename(columns={col: f"{col}_2020" for col in simd_value_cols})

    return simd_2016, simd_2020


def build_context_panel_2011_native(
    population_panel: pd.DataFrame,
    simd_panel: pd.DataFrame,
) -> pd.DataFrame:
    simd_2016_wide, simd_2020_wide = build_simd_wide(simd_panel)

    context_df = population_panel.merge(
        simd_2016_wide,
        on="dz_code_2011",
        how="left",
        validate="m:1",
    )

    context_df = context_df.merge(
        simd_2020_wide,
        on="dz_code_2011",
        how="left",
        validate="m:1",
    )

    context_df["period_group"] = context_df["year"].apply(assign_period_group)
    context_df["simd_reference_year"] = context_df["year"].apply(
        lambda y: 2016 if y <= 2019 else 2020
    )

    return context_df


def build_zone_year_context_2022(
    context_panel_2011_native: pd.DataFrame,
    bridge_df: pd.DataFrame,
    zones_master_2022: pd.DataFrame,
) -> pd.DataFrame:
    # Keep only needed bridge fields
    bridge_use = bridge_df[
        [
            "dz_code_2011",
            "dz_code_2022",
            "pct_of_2011",
            "pct_of_2022",
        ]
    ].copy()

    merged = context_panel_2011_native.merge(
        bridge_use,
        on="dz_code_2011",
        how="inner",
        validate="m:m",
    )

    # Allocate additive counts using source-zone share
    merged["population_total_allocated"] = merged["population_total"] * merged["pct_of_2011"]
    merged["older_population_65_plus_allocated"] = (
        merged["older_population_65_plus"] * merged["pct_of_2011"]
    )

    simd_cols_2016 = [
        "simd_rank_overall_2016",
        "income_domain_rank_2016",
        "employment_domain_rank_2016",
        "health_domain_rank_2016",
        "education_domain_rank_2016",
        "housing_domain_rank_2016",
        "access_domain_rank_2016",
        "crime_domain_rank_2016",
    ]

    simd_cols_2020 = [
        "simd_rank_overall_2020",
        "income_domain_rank_2020",
        "employment_domain_rank_2020",
        "health_domain_rank_2020",
        "education_domain_rank_2020",
        "housing_domain_rank_2020",
        "access_domain_rank_2020",
        "crime_domain_rank_2020",
    ]

    all_simd_cols = simd_cols_2016 + simd_cols_2020

    group_keys = ["dz_code_2022", "year"]

    merged["target_weight_norm"] = (
        merged["pct_of_2022"] / merged.groupby(group_keys)["pct_of_2022"].transform("sum")
    )

    for col in all_simd_cols:
        merged[f"weighted_{col}"] = merged[col] * merged["target_weight_norm"]

    agg_dict = {
        "population_total_allocated": "sum",
        "older_population_65_plus_allocated": "sum",
        "period_group": "first",
        "simd_reference_year": "first",
    }

    for col in all_simd_cols:
        agg_dict[f"weighted_{col}"] = "sum"

    zone_year_context = merged.groupby(group_keys, as_index=False).agg(agg_dict)

    zone_year_context = zone_year_context.rename(
        columns={
            "population_total_allocated": "population_total",
            "older_population_65_plus_allocated": "older_population_65_plus",
        }
    )

    zone_year_context["older_population_share"] = (
        zone_year_context["older_population_65_plus"] / zone_year_context["population_total"]
    )

    for col in all_simd_cols:
        zone_year_context = zone_year_context.rename(
            columns={f"weighted_{col}": col}
        )

    # Merge stable 2022 zone metadata
    zone_metadata_cols = [
        "dz_code_2022",
        "dz_name_2022",
        "ur6_name",
        "ur8_name",
        "is_rural",
        "is_accessible_rural",
        "is_remote_rural",
    ]
    zone_metadata = zones_master_2022[zone_metadata_cols].copy()

    zone_year_context = zone_year_context.merge(
        zone_metadata,
        on="dz_code_2022",
        how="left",
        validate="m:1",
    )

    # Build active SIMD columns based on reference year
    active_base_names = [
        "simd_rank_overall",
        "income_domain_rank",
        "employment_domain_rank",
        "health_domain_rank",
        "education_domain_rank",
        "housing_domain_rank",
        "access_domain_rank",
        "crime_domain_rank",
    ]

    mask_2016 = zone_year_context["simd_reference_year"] == 2016

    for base in active_base_names:
        zone_year_context[f"active_{base}"] = np.where(
            mask_2016,
            zone_year_context[f"{base}_2016"],
            zone_year_context[f"{base}_2020"],
        )

    final_col_order = [
        "dz_code_2022",
        "dz_name_2022",
        "year",
        "period_group",
        "simd_reference_year",
        "ur6_name",
        "ur8_name",
        "is_rural",
        "is_accessible_rural",
        "is_remote_rural",
        "population_total",
        "older_population_65_plus",
        "older_population_share",
        "simd_rank_overall_2016",
        "income_domain_rank_2016",
        "employment_domain_rank_2016",
        "health_domain_rank_2016",
        "education_domain_rank_2016",
        "housing_domain_rank_2016",
        "access_domain_rank_2016",
        "crime_domain_rank_2016",
        "simd_rank_overall_2020",
        "income_domain_rank_2020",
        "employment_domain_rank_2020",
        "health_domain_rank_2020",
        "education_domain_rank_2020",
        "housing_domain_rank_2020",
        "access_domain_rank_2020",
        "crime_domain_rank_2020",
        "active_simd_rank_overall",
        "active_income_domain_rank",
        "active_employment_domain_rank",
        "active_health_domain_rank",
        "active_education_domain_rank",
        "active_housing_domain_rank",
        "active_access_domain_rank",
        "active_crime_domain_rank",
    ]

    zone_year_context = zone_year_context[final_col_order].copy()

    return zone_year_context


def main() -> None:
    print("Starting context-data build...")

    context_interim_dir = INTERIM_DIR / "context"
    context_interim_dir.mkdir(parents=True, exist_ok=True)

    context_processed_dir = PROCESSED_DIR / "context"
    context_processed_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Build cleaned population panel
    # -------------------------
    print(f"\nPopulation root folder: {POPULATION_DIR}")
    population_files = sorted(POPULATION_DIR.rglob("*.xlsx"))

    if not population_files:
        raise FileNotFoundError("No population Excel files were found.")

    valid_year_files = []
    for file_path in population_files:
        year = extract_year_from_name(file_path)
        if year is not None:
            valid_year_files.append((file_path, year))

    valid_year_files = sorted(valid_year_files, key=lambda x: x[1])

    cleaned_population_frames = []
    for file_path, year in valid_year_files:
        print(f"\nCleaning population file for year {year}: {file_path.name}")
        cleaned_df = clean_population_file(file_path, year)
        print(f"Rows loaded for {year}: {len(cleaned_df)}")
        cleaned_population_frames.append(cleaned_df)

    population_panel = pd.concat(cleaned_population_frames, ignore_index=True)
    population_output_csv = context_interim_dir / "population_panel_2011_geography.csv"
    population_panel.to_csv(population_output_csv, index=False)

    print("\n--- Cleaned Population Panel Summary ---")
    print(f"Total rows: {len(population_panel)}")
    print(f"Unique years: {sorted(population_panel['year'].unique().tolist())}")
    print(f"Unique 2011 Data Zones: {population_panel['dz_code_2011'].nunique()}")

    # -------------------------
    # Build cleaned SIMD panel
    # -------------------------
    print("\nCleaning SIMD 2016...")
    simd_2016 = clean_simd_2016(SIMD_2016_FILE)
    print(f"SIMD 2016 rows: {len(simd_2016)}")

    print("\nCleaning SIMD 2020...")
    simd_2020 = clean_simd_2020(SIMD_2020_FILE)
    print(f"SIMD 2020 rows: {len(simd_2020)}")

    simd_panel = pd.concat([simd_2016, simd_2020], ignore_index=True)
    simd_output_csv = context_interim_dir / "simd_panel_2011_geography.csv"
    simd_panel.to_csv(simd_output_csv, index=False)

    print("\n--- Cleaned SIMD Panel Summary ---")
    print(f"Total rows: {len(simd_panel)}")
    print(f"SIMD years: {sorted(simd_panel['simd_year'].unique().tolist())}")
    print(f"Unique 2011 Data Zones: {simd_panel['dz_code_2011'].nunique()}")

    # -------------------------
    # Build native 2011 context layer
    # -------------------------
    print("\nBuilding native 2011 context layer...")
    context_panel_2011_native = build_context_panel_2011_native(population_panel, simd_panel)

    context_native_output_csv = context_interim_dir / "context_panel_2011_native.csv"
    context_panel_2011_native.to_csv(context_native_output_csv, index=False)

    print("\n--- Native 2011 Context Panel Summary ---")
    print(f"Total rows: {len(context_panel_2011_native)}")
    print(f"Unique years: {sorted(context_panel_2011_native['year'].unique().tolist())}")
    print(f"Unique 2011 Data Zones: {context_panel_2011_native['dz_code_2011'].nunique()}")

    # -------------------------
    # Build harmonised 2022 zone_year_context
    # -------------------------
    print("\nLoading bridge and 2022 zone metadata...")
    bridge_csv = INTERIM_DIR / "geography_bridge" / "zone_lookup_bridge.csv"
    zones_master_2022_gpkg = PROCESSED_DIR / "geography" / "zones_master_2022.gpkg"

    bridge_df = pd.read_csv(bridge_csv)
    zones_master_2022 = gpd.read_file(zones_master_2022_gpkg, layer="zones_master_2022")

    print(f"Bridge rows loaded: {len(bridge_df)}")
    print(f"2022 geography rows loaded: {len(zones_master_2022)}")

    print("\nBuilding harmonised 2022 zone_year_context...")
    zone_year_context_2022 = build_zone_year_context_2022(
        context_panel_2011_native=context_panel_2011_native,
        bridge_df=bridge_df,
        zones_master_2022=zones_master_2022,
    )

    zone_year_context_output_csv = context_processed_dir / "zone_year_context_2022.csv"
    zone_year_context_2022.to_csv(zone_year_context_output_csv, index=False)

    missing_active_simd = zone_year_context_2022["active_simd_rank_overall"].isna().sum()
    missing_population = zone_year_context_2022["population_total"].isna().sum()

    print("\n--- Harmonised 2022 zone_year_context Summary ---")
    print(f"Total rows: {len(zone_year_context_2022)}")
    print(f"Unique years: {sorted(zone_year_context_2022['year'].unique().tolist())}")
    print(f"Unique 2022 Data Zones: {zone_year_context_2022['dz_code_2022'].nunique()}")

    print("\nRows by year:")
    print(zone_year_context_2022.groupby("year").size())

    print("\nRows by period_group:")
    print(zone_year_context_2022.groupby("period_group").size())

    print("\nRows by is_rural:")
    print(zone_year_context_2022.groupby("is_rural").size())

    print(f"\nMissing active SIMD overall-rank values: {missing_active_simd}")
    print(f"Missing population_total values: {missing_population}")

    print("\nPreview of harmonised 2022 zone_year_context:")
    print(zone_year_context_2022.head())

    print(f"\nzone_year_context output file: {zone_year_context_output_csv}")
    print("\nContext-data build completed successfully.")


if __name__ == "__main__":
    main()