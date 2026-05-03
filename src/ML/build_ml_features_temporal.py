import numpy as np
import pandas as pd

from config.paths import PROCESSED_DIR


TARGET_COL = "bank_access_deterioration_flag"
SEVERE_TARGET_COL = "bank_access_severe_deterioration_flag"


def validate_required_columns(df: pd.DataFrame, required_cols: list[str], df_name: str) -> None:
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {df_name}: {missing}")


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return np.where(denominator > 0, numerator / denominator, np.nan)


def main() -> None:
    print("Starting ml_features_temporal_2022 build...")

    processed_dir = PROCESSED_DIR
    ml_dir = processed_dir / "ml"
    ml_dir.mkdir(parents=True, exist_ok=True)

    context_path = processed_dir / "context" / "zone_year_context_2022.csv"
    baseline_access_path = processed_dir / "accessibility" / "zone_accessibility_baseline_2022.csv"
    temporal_access_path = processed_dir / "accessibility" / "bank_accessibility_temporal_2019_2023.csv"
    closure_change_path = processed_dir / "bank_closures" / "bank_closure_change_features_2022.csv"
    underserved_labels_path = processed_dir / "accessibility" / "underserved_labels_baseline_2022.csv"

    print(f"Loading context from: {context_path}")
    context = pd.read_csv(context_path)

    print(f"Loading baseline accessibility from: {baseline_access_path}")
    baseline_access = pd.read_csv(baseline_access_path)

    print(f"Loading temporal bank accessibility from: {temporal_access_path}")
    temporal_access = pd.read_csv(temporal_access_path)

    print(f"Loading closure change features from: {closure_change_path}")
    closure_change = pd.read_csv(closure_change_path)

    print(f"Loading underserved labels from: {underserved_labels_path}")
    underserved = pd.read_csv(underserved_labels_path)

    print(f"Context rows loaded: {len(context)}")
    print(f"Baseline accessibility rows loaded: {len(baseline_access)}")
    print(f"Temporal accessibility rows loaded: {len(temporal_access)}")
    print(f"Closure change rows loaded: {len(closure_change)}")
    print(f"Underserved label rows loaded: {len(underserved)}")

    # Use latest harmonised context year for V2 zone-level modelling
    context_latest = context[context["year"] == 2023].copy()
    print(f"\nRows after filtering context to 2023: {len(context_latest)}")

    context_keep_cols = [
        "dz_code_2022",
        "dz_name_2022",
        "ur6_name",
        "ur8_name",
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
    ]
    context_keep_cols = [col for col in context_keep_cols if col in context_latest.columns]
    context_latest = context_latest[context_keep_cols].drop_duplicates(subset=["dz_code_2022"]).copy()

    baseline_keep_cols = [
        "dz_code_2022",
        "dist_to_nearest_bank_m",
        "dist_to_nearest_atm_m",
        "dist_to_nearest_post_office_m",
        "dist_to_nearest_any_access_point_m",
        "dist_to_nearest_bank_km",
        "dist_to_nearest_atm_km",
        "dist_to_nearest_post_office_km",
        "dist_to_nearest_any_access_point_km",
    ]
    validate_required_columns(baseline_access, ["dz_code_2022"], "zone_accessibility_baseline_2022")
    baseline_keep_cols = [col for col in baseline_keep_cols if col in baseline_access.columns]
    baseline_access = baseline_access[baseline_keep_cols].drop_duplicates(subset=["dz_code_2022"]).copy()

    temporal_keep_cols = [
        "dz_code_2022",
        "nearest_bank_id_2019",
        "nearest_bank_brand_2019",
        "nearest_bank_type_2019",
        "nearest_bank_town_2019",
        "dist_to_nearest_bank_m_2019",
        "dist_to_nearest_bank_km_2019",
        "nearest_bank_id_2023",
        "nearest_bank_brand_2023",
        "nearest_bank_type_2023",
        "nearest_bank_town_2023",
        "dist_to_nearest_bank_m_2023",
        "dist_to_nearest_bank_km_2023",
        "bank_distance_change_km_post_minus_pre",
        "bank_distance_change_m_post_minus_pre",
        "bank_distance_pct_change_post_vs_pre",
        "bank_access_deterioration_flag",
        "bank_access_improvement_flag",
        "bank_access_no_change_flag",
        "bank_access_major_deterioration_flag",
        "bank_access_severe_deterioration_flag",
    ]
    validate_required_columns(
        temporal_access,
        ["dz_code_2022", "bank_distance_change_km_post_minus_pre", TARGET_COL],
        "bank_accessibility_temporal_2019_2023",
    )
    temporal_keep_cols = [col for col in temporal_keep_cols if col in temporal_access.columns]
    temporal_access = temporal_access[temporal_keep_cols].drop_duplicates(subset=["dz_code_2022"]).copy()

    closure_keep_cols = [
        "dz_code_2022",
        "pre_covid_closures_total",
        "pre_covid_years_with_closure",
        "pre_covid_unique_brands_total",
        "pre_covid_unique_branch_types_total",
        "pre_covid_year_count",
        "pre_covid_closures_annual_rate",
        "covid_transition_closures_total",
        "covid_transition_years_with_closure",
        "covid_transition_unique_brands_total",
        "covid_transition_unique_branch_types_total",
        "covid_transition_year_count",
        "covid_transition_closures_annual_rate",
        "post_covid_closures_total",
        "post_covid_years_with_closure",
        "post_covid_unique_brands_total",
        "post_covid_unique_branch_types_total",
        "post_covid_year_count",
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
    ]
    validate_required_columns(closure_change, ["dz_code_2022"], "bank_closure_change_features_2022")
    closure_keep_cols = [col for col in closure_keep_cols if col in closure_change.columns]
    closure_change = closure_change[closure_keep_cols].drop_duplicates(subset=["dz_code_2022"]).copy()

    underserved_keep_cols = [
        "dz_code_2022",
        "underserved_baseline",
        "critical_underserved_baseline",
    ]
    validate_required_columns(underserved, underserved_keep_cols, "underserved_labels_baseline_2022")
    underserved = underserved[underserved_keep_cols].drop_duplicates(subset=["dz_code_2022"]).copy()

    merged = context_latest.merge(
        baseline_access,
        on="dz_code_2022",
        how="inner",
        validate="1:1",
    )

    merged = merged.merge(
        temporal_access,
        on="dz_code_2022",
        how="inner",
        validate="1:1",
    )

    merged = merged.merge(
        closure_change,
        on="dz_code_2022",
        how="inner",
        validate="1:1",
    )

    merged = merged.merge(
        underserved,
        on="dz_code_2022",
        how="left",
        validate="1:1",
    )

    merged["underserved_baseline"] = merged["underserved_baseline"].fillna(0).astype(int)
    merged["critical_underserved_baseline"] = merged["critical_underserved_baseline"].fillna(0).astype(int)

    # Derived temporal features
    if "dist_to_nearest_bank_km" in merged.columns and "dist_to_nearest_bank_km_2023" in merged.columns:
        merged["current_vs_2023_bank_distance_gap_km"] = (
            merged["dist_to_nearest_bank_km"] - merged["dist_to_nearest_bank_km_2023"]
        )

    if "dist_to_nearest_bank_km_2023" in merged.columns and "dist_to_nearest_bank_km_2019" in merged.columns:
        merged["bank_distance_ratio_2023_to_2019"] = safe_ratio(
            merged["dist_to_nearest_bank_km_2023"],
            merged["dist_to_nearest_bank_km_2019"],
        )

    if "post_covid_closures_total" in merged.columns and "pre_covid_closures_total" in merged.columns:
        merged["post_minus_pre_closures_total"] = (
            merged["post_covid_closures_total"] - merged["pre_covid_closures_total"]
        )

    if "post_covid_closures_annual_rate" in merged.columns and "pre_covid_closures_annual_rate" in merged.columns:
        merged["post_minus_pre_closure_rate"] = (
            merged["post_covid_closures_annual_rate"] - merged["pre_covid_closures_annual_rate"]
        )

    if "post_covid_closures_total" in merged.columns and "population_total" in merged.columns:
        merged["post_covid_closures_per_1000_pop"] = np.where(
            merged["population_total"] > 0,
            (merged["post_covid_closures_total"] / merged["population_total"]) * 1000,
            np.nan,
        )

    if "cumulative_closures_to_2023" in merged.columns and "population_total" in merged.columns:
        merged["cumulative_closures_per_1000_pop"] = np.where(
            merged["population_total"] > 0,
            (merged["cumulative_closures_to_2023"] / merged["population_total"]) * 1000,
            np.nan,
        )

    if "bank_distance_change_km_post_minus_pre" in merged.columns and "post_covid_closures_total" in merged.columns:
        merged["bank_change_x_post_closures"] = (
            merged["bank_distance_change_km_post_minus_pre"] * merged["post_covid_closures_total"]
        )

    if "bank_distance_change_km_post_minus_pre" in merged.columns and "active_access_domain_rank_severity" in merged.columns:
        merged["bank_change_x_access_severity"] = (
            merged["bank_distance_change_km_post_minus_pre"] * merged["active_access_domain_rank_severity"]
        )

    # Clean rank-ratio infinities if any
    merged = merged.replace([np.inf, -np.inf], np.nan)

    print("\n--- Temporal ML Feature Summary ---")
    print(f"Rows after full merge: {len(merged)}")
    print(f"Columns after full merge: {len(merged.columns)}")

    print("\nTarget distribution:")
    print(merged[TARGET_COL].value_counts(dropna=False))

    print("\nSevere target distribution:")
    if SEVERE_TARGET_COL in merged.columns:
        print(merged[SEVERE_TARGET_COL].value_counts(dropna=False))

    print("\nRows by rural flag:")
    print(merged["is_rural"].value_counts(dropna=False))

    print("\nRows by UR6:")
    print(merged["ur6_name"].value_counts(dropna=False))

    preview_cols = [
        "dz_code_2022",
        "dz_name_2022",
        "ur6_name",
        "ur8_name",
        "dist_to_nearest_bank_km_2019",
        "dist_to_nearest_bank_km_2023",
        "bank_distance_change_km_post_minus_pre",
        "post_covid_closures_total",
        "closure_rate_change_post_minus_pre",
        "bank_access_deterioration_flag",
        "bank_access_major_deterioration_flag",
        "bank_access_severe_deterioration_flag",
    ]
    preview_cols = [col for col in preview_cols if col in merged.columns]

    print("\nPreview of temporal ML master table:")
    print(merged[preview_cols].head(10))

    output_path = ml_dir / "ml_features_temporal_2022.csv"
    merged.to_csv(output_path, index=False)

    print("\nml_features_temporal_2022 build completed successfully.")
    print(f"Output CSV: {output_path}")


if __name__ == "__main__":
    main()