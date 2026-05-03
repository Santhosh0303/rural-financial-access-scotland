import numpy as np
import pandas as pd

from config.paths import PROCESSED_DIR


TARGET_COL = "bank_access_deterioration_flag"
AUX_TARGET_COLS = [
    "bank_access_major_deterioration_flag",
    "bank_access_severe_deterioration_flag",
]
REFERENCE_YEAR = 2023


def unique_existing(columns: list[str], df: pd.DataFrame) -> list[str]:
    seen = set()
    output = []
    for col in columns:
        if col in df.columns and col not in seen:
            output.append(col)
            seen.add(col)
    return output


def validate_required_columns(df: pd.DataFrame, required_cols: list[str], df_name: str) -> None:
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {df_name}: {missing}")


def add_temporal_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "first_closure_year" in out.columns:
        out["has_first_closure_year"] = out["first_closure_year"].notna().astype(int)
        out["years_since_first_closure_to_2023"] = np.where(
            out["first_closure_year"].notna(),
            REFERENCE_YEAR - out["first_closure_year"],
            0,
        )
    else:
        out["has_first_closure_year"] = 0
        out["years_since_first_closure_to_2023"] = 0

    if "cumulative_closures_to_2023" in out.columns:
        out["has_any_closure_to_2023"] = (out["cumulative_closures_to_2023"] > 0).astype(int)

    if (
        "post_covid_closures_total" in out.columns
        and "pre_covid_closures_total" in out.columns
    ):
        out["post_vs_pre_closure_ratio"] = np.where(
            out["pre_covid_closures_total"] > 0,
            out["post_covid_closures_total"] / out["pre_covid_closures_total"],
            np.where(out["post_covid_closures_total"] > 0, 1.0, 0.0),
        )

    if (
        "post_covid_closures_annual_rate" in out.columns
        and "pre_covid_closures_annual_rate" in out.columns
    ):
        out["post_vs_pre_closure_rate_ratio"] = np.where(
            out["pre_covid_closures_annual_rate"] > 0,
            out["post_covid_closures_annual_rate"] / out["pre_covid_closures_annual_rate"],
            np.where(out["post_covid_closures_annual_rate"] > 0, 1.0, 0.0),
        )

    if (
        "dist_to_nearest_bank_km_2019" in out.columns
        and "population_total" in out.columns
    ):
        out["bank_km_2019_x_population"] = (
            out["dist_to_nearest_bank_km_2019"] * out["population_total"]
        )

    if (
        "dist_to_nearest_bank_km_2019" in out.columns
        and "older_population_share" in out.columns
    ):
        out["bank_km_2019_x_older_share"] = (
            out["dist_to_nearest_bank_km_2019"] * out["older_population_share"]
        )

    return out


def build_encoded_table(
    readable_df: pd.DataFrame,
    id_cols: list[str],
    categorical_cols: list[str],
    target_cols: list[str],
) -> pd.DataFrame:
    encoded = readable_df.copy()

    drop_cols = [col for col in id_cols if col in encoded.columns]
    encoded = encoded.drop(columns=drop_cols, errors="ignore")

    numeric_cols = [
        col for col in encoded.columns
        if col not in categorical_cols and col not in target_cols
    ]
    numeric_cols = [col for col in numeric_cols if pd.api.types.is_numeric_dtype(encoded[col])]

    for col in numeric_cols:
        encoded[col] = pd.to_numeric(encoded[col], errors="coerce")

    encoded[numeric_cols] = encoded[numeric_cols].replace([np.inf, -np.inf], np.nan)
    encoded[numeric_cols] = encoded[numeric_cols].fillna(0)

    existing_cat_cols = [col for col in categorical_cols if col in encoded.columns]
    for col in existing_cat_cols:
        encoded[col] = encoded[col].astype("string").fillna("Unknown").str.strip()
        encoded[col] = encoded[col].replace("", "Unknown")

    encoded = pd.get_dummies(
        encoded,
        columns=existing_cat_cols,
        drop_first=False,
        dtype=int,
    )

    constant_cols = []
    protected_cols = [col for col in target_cols if col in encoded.columns]
    for col in encoded.columns:
        if col in protected_cols:
            continue
        if encoded[col].nunique(dropna=False) <= 1:
            constant_cols.append(col)

    if constant_cols:
        encoded = encoded.drop(columns=constant_cols, errors="ignore")

    return encoded


def main() -> None:
    print("Starting build_model_ready_temporal...")

    ml_dir = PROCESSED_DIR / "ml"
    input_path = ml_dir / "ml_features_temporal_2022.csv"

    print(f"Loading temporal ML master table from: {input_path}")
    df = pd.read_csv(input_path)

    print(f"Rows loaded: {len(df)}")
    print(f"Columns loaded: {len(df.columns)}")

    validate_required_columns(df, ["dz_code_2022", TARGET_COL, "is_rural"], "ml_features_temporal_2022")

    df = add_temporal_derived_features(df)

    id_cols = unique_existing(
        [
            "dz_code_2022",
            "dz_name_2022",
            "nearest_bank_id_2019",
            "nearest_bank_brand_2019",
            "nearest_bank_town_2019",
            "nearest_bank_id_2023",
            "nearest_bank_brand_2023",
            "nearest_bank_town_2023",
        ],
        df,
    )

    target_cols = unique_existing([TARGET_COL] + AUX_TARGET_COLS, df)

    categorical_cols = unique_existing(
        [
            "ur6_name",
            "ur8_name",
            "nearest_bank_type_2019",
        ],
        df,
    )

    # Keep only non-leaky explanatory features
    feature_cols = unique_existing(
        [
            "is_rural",
            "is_accessible_rural",
            "is_remote_rural",

            "population_total",
            "older_population_65_plus",
            "older_population_share",

            "active_simd_rank_overall",
            "active_income_domain_rank",
            "active_employment_domain_rank",
            "active_health_domain_rank",
            "active_education_domain_rank",
            "active_access_domain_rank",
            "active_crime_domain_rank",

            "active_simd_rank_overall_severity",
            "active_income_domain_rank_severity",
            "active_employment_domain_rank_severity",
            "active_health_domain_rank_severity",
            "active_education_domain_rank_severity",
            "active_access_domain_rank_severity",
            "active_crime_domain_rank_severity",

            # cross-service current context
            "dist_to_nearest_atm_km",
            "dist_to_nearest_post_office_km",
            "dist_to_nearest_any_access_point_km",

            # baseline bank state before deterioration window
            "dist_to_nearest_bank_km_2019",
            "nearest_bank_type_2019",

            # closure-history features
            "pre_covid_closures_total",
            "pre_covid_years_with_closure",
            "pre_covid_unique_brands_total",
            "pre_covid_unique_branch_types_total",
            "pre_covid_closures_annual_rate",

            "covid_transition_closures_total",
            "covid_transition_years_with_closure",
            "covid_transition_unique_brands_total",
            "covid_transition_unique_branch_types_total",
            "covid_transition_closures_annual_rate",

            "post_covid_closures_total",
            "post_covid_years_with_closure",
            "post_covid_unique_brands_total",
            "post_covid_unique_branch_types_total",
            "post_covid_closures_annual_rate",

            "first_closure_year",
            "cumulative_closures_to_2023",
            "any_pre_covid_closure_flag",
            "any_post_covid_closure_flag",
            "post_covid_only_closure_flag",
            "closure_rate_change_post_minus_pre",
            "closure_rate_ratio_post_to_pre",
            "closure_count_change_post_minus_pre",
            "closure_deterioration_flag",
            "closure_persistence_flag",

            # derived non-leaky temporal features
            "has_first_closure_year",
            "years_since_first_closure_to_2023",
            "has_any_closure_to_2023",
            "post_minus_pre_closures_total",
            "post_minus_pre_closure_rate",
            "post_vs_pre_closure_ratio",
            "post_vs_pre_closure_rate_ratio",
            "post_covid_closures_per_1000_pop",
            "cumulative_closures_per_1000_pop",
            "bank_km_2019_x_population",
            "bank_km_2019_x_older_share",

            # comparator labels kept only in readable tables
            "underserved_baseline",
            "critical_underserved_baseline",
        ],
        df,
    )

    # Explicitly remove outcome-leaking / post-outcome columns from features
    leakage_cols = [
        "dist_to_nearest_bank_m_2023",
        "dist_to_nearest_bank_km_2023",
        "bank_distance_change_km_post_minus_pre",
        "bank_distance_change_m_post_minus_pre",
        "bank_distance_pct_change_post_vs_pre",
        "bank_access_improvement_flag",
        "bank_access_no_change_flag",
        "bank_access_major_deterioration_flag",
        "bank_access_severe_deterioration_flag",
        "current_vs_2023_bank_distance_gap_km",
        "bank_distance_ratio_2023_to_2019",
        "bank_change_x_post_closures",
        "bank_change_x_access_severity",
    ]
    feature_cols = [col for col in feature_cols if col not in leakage_cols]

    readable_cols = unique_existing(id_cols + categorical_cols + feature_cols + target_cols, df)
    readable_df = df[readable_cols].copy()

    all_zone_readable = readable_df.copy()
    rural_readable = readable_df[readable_df["is_rural"] == 1].copy()

    all_zone_encoded = build_encoded_table(
        readable_df=all_zone_readable,
        id_cols=id_cols,
        categorical_cols=categorical_cols,
        target_cols=target_cols,
    )

    rural_encoded = build_encoded_table(
        readable_df=rural_readable,
        id_cols=id_cols,
        categorical_cols=categorical_cols,
        target_cols=target_cols,
    )

    print("\n--- Model Ready Temporal Summary ---")
    print(f"All-zone readable rows: {len(all_zone_readable)}")
    print(f"All-zone encoded rows: {len(all_zone_encoded)}")
    print(f"All-zone encoded columns: {len(all_zone_encoded.columns)}")

    print(f"\nRural-only readable rows: {len(rural_readable)}")
    print(f"Rural-only encoded rows: {len(rural_encoded)}")
    print(f"Rural-only encoded columns: {len(rural_encoded.columns)}")

    print("\nAll-zone target distribution:")
    print(all_zone_readable[TARGET_COL].value_counts(dropna=False))

    print("\nRural-only target distribution:")
    print(rural_readable[TARGET_COL].value_counts(dropna=False))

    if "bank_access_severe_deterioration_flag" in all_zone_readable.columns:
        print("\nAll-zone severe target distribution:")
        print(all_zone_readable["bank_access_severe_deterioration_flag"].value_counts(dropna=False))

        print("\nRural-only severe target distribution:")
        print(rural_readable["bank_access_severe_deterioration_flag"].value_counts(dropna=False))

    preview_cols = unique_existing(
        [
            "dz_code_2022",
            "dz_name_2022",
            "ur6_name",
            "ur8_name",
            "dist_to_nearest_bank_km_2019",
            "pre_covid_closures_total",
            "post_covid_closures_total",
            "closure_rate_change_post_minus_pre",
            "bank_access_deterioration_flag",
            "bank_access_major_deterioration_flag",
            "bank_access_severe_deterioration_flag",
        ],
        all_zone_readable,
    )

    print("\nPreview of all-zone readable table:")
    print(all_zone_readable[preview_cols].head(10))

    print("\nPreview of rural-only encoded table:")
    print(rural_encoded.head(10))

    all_readable_path = ml_dir / "model_ready_temporal_all_readable.csv"
    all_encoded_path = ml_dir / "model_ready_temporal_all_encoded.csv"
    rural_readable_path = ml_dir / "model_ready_temporal_rural_readable.csv"
    rural_encoded_path = ml_dir / "model_ready_temporal_rural_encoded.csv"

    all_zone_readable.to_csv(all_readable_path, index=False)
    all_zone_encoded.to_csv(all_encoded_path, index=False)
    rural_readable.to_csv(rural_readable_path, index=False)
    rural_encoded.to_csv(rural_encoded_path, index=False)

    print("\nbuild_model_ready_temporal completed successfully.")
    print(f"All-zone readable output CSV: {all_readable_path}")
    print(f"All-zone encoded output CSV: {all_encoded_path}")
    print(f"Rural-only readable output CSV: {rural_readable_path}")
    print(f"Rural-only encoded output CSV: {rural_encoded_path}")


if __name__ == "__main__":
    main()