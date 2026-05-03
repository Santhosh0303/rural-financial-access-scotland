from pathlib import Path
import pandas as pd

from config.paths import PROCESSED_DIR


TARGET_COL = "underserved_baseline"

# These columns directly create or strongly leak the target.
LEAKAGE_COLS = [
    "dist_to_nearest_bank_m",
    "dist_to_nearest_atm_m",
    "dist_to_nearest_post_office_m",
    "dist_to_nearest_any_access_point_m",
    "dist_to_nearest_bank_km",
    "dist_to_nearest_atm_km",
    "dist_to_nearest_post_office_km",
    "dist_to_nearest_any_access_point_km",
    "flag_far_bank",
    "flag_far_atm",
    "flag_far_post_office",
    "flag_far_any_access_point",
    "service_gap_count",
    "total_gap_score",
    "dominant_gap_type",
    "critical_underserved_baseline",
    "bank_vs_any_ratio",
    "atm_vs_any_ratio",
    "post_vs_any_ratio",
    "bank_minus_any_m",
    "atm_minus_any_m",
    "post_minus_any_m",
]

# Non-feature identifier / descriptive columns
NON_FEATURE_COLS = [
    "dz_code_2022",
    "dz_name_2022",
]

# Categorical predictors we will one-hot encode
CATEGORICAL_COLS = [
    "ur6_name",
    "ur8_name",
]

# Numeric predictors that are safe to keep
NUMERIC_FEATURE_COLS = [
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
    "active_housing_domain_rank",
    "active_access_domain_rank",
    "active_crime_domain_rank",
    "active_simd_rank_overall_severity",
    "active_income_domain_rank_severity",
    "active_employment_domain_rank_severity",
    "active_health_domain_rank_severity",
    "active_education_domain_rank_severity",
    "active_housing_domain_rank_severity",
    "active_access_domain_rank_severity",
    "active_crime_domain_rank_severity",
]


def validate_required_columns(df: pd.DataFrame, cols: list[str], df_name: str) -> None:
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {df_name}: {missing}")


def build_model_ready_table(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build a rural-only, non-leaky baseline modelling table.
    Returns:
    - encoded model-ready table
    - human-readable pre-encoded table
    """
    out = df.copy()

    # Keep only rural zones for first modelling stage
    out = out[out["is_rural"] == 1].copy()

    if out.empty:
        raise ValueError("No rural rows found after filtering.")

    # Check required columns
    validate_required_columns(
        out,
        [TARGET_COL] + CATEGORICAL_COLS + NUMERIC_FEATURE_COLS,
        "ml_features_baseline_2022",
    )

    # Human-readable version first
    readable_cols = (
        NON_FEATURE_COLS
        + ["is_rural"]
        + CATEGORICAL_COLS
        + NUMERIC_FEATURE_COLS
        + [TARGET_COL, "critical_underserved_baseline"]
    )
    readable_cols = [col for col in readable_cols if col in out.columns]
    readable_df = out[readable_cols].copy()

    # Strict predictor frame: only safe features + target
    predictor_cols = CATEGORICAL_COLS + NUMERIC_FEATURE_COLS
    predictor_cols = [col for col in predictor_cols if col in out.columns]

    model_df = out[predictor_cols + [TARGET_COL]].copy()

    # One-hot encode categorical predictors
    model_df = pd.get_dummies(
        model_df,
        columns=[col for col in CATEGORICAL_COLS if col in model_df.columns],
        drop_first=False,
        dtype=int,
    )

    # Final safety checks
    if TARGET_COL not in model_df.columns:
        raise ValueError(f"Target column {TARGET_COL} missing after transformation.")

    if model_df[TARGET_COL].isna().any():
        raise ValueError(f"Target column {TARGET_COL} contains missing values.")

    # Make sure obviously leaky columns are not present
    present_leaks = [col for col in LEAKAGE_COLS if col in model_df.columns]
    if present_leaks:
        raise ValueError(f"Leaky columns still present in model table: {present_leaks}")

    return model_df, readable_df


def main() -> None:
    print("Starting build_model_ready_baseline...")

    ml_dir = PROCESSED_DIR / "ml"
    source_path = ml_dir / "ml_features_baseline_2022.csv"

    print(f"Loading ML master table from: {source_path}")
    ml_features = pd.read_csv(source_path)

    print(f"Rows loaded: {len(ml_features)}")
    print(f"Columns loaded: {len(ml_features.columns)}")

    model_ready_df, readable_df = build_model_ready_table(ml_features)

    target_counts = model_ready_df[TARGET_COL].value_counts(dropna=False).sort_index()

    print("\n--- Model Ready Summary ---")
    print(f"Readable rural rows: {len(readable_df)}")
    print(f"Model-ready rural rows: {len(model_ready_df)}")
    print(f"Model-ready columns: {len(model_ready_df.columns)}")

    print("\nTarget distribution in rural-only model table:")
    print(target_counts)

    print("\nPositive class share:")
    print(model_ready_df[TARGET_COL].mean())

    print("\nPreview of readable rural table:")
    print(readable_df.head())

    print("\nPreview of encoded model-ready table:")
    print(model_ready_df.head())

    output_readable_csv = ml_dir / "model_ready_baseline_rural_readable.csv"
    output_model_csv = ml_dir / "model_ready_baseline_rural_encoded.csv"

    readable_df.to_csv(output_readable_csv, index=False)
    model_ready_df.to_csv(output_model_csv, index=False)

    print("\nbuild_model_ready_baseline completed successfully.")
    print(f"Readable output CSV: {output_readable_csv}")
    print(f"Encoded model-ready output CSV: {output_model_csv}")


if __name__ == "__main__":
    main()