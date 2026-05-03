from pathlib import Path
import pandas as pd

from config.paths import PROCESSED_DIR


CONTEXT_KEEP_COLS = [
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
    "active_housing_domain_rank",
    "active_access_domain_rank",
    "active_crime_domain_rank",
]

ACCESS_KEEP_COLS = [
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

LABEL_KEEP_COLS = [
    "dz_code_2022",
    "flag_far_bank",
    "flag_far_atm",
    "flag_far_post_office",
    "flag_far_any_access_point",
    "service_gap_count",
    "total_gap_score",
    "dominant_gap_type",
    "underserved_baseline",
    "critical_underserved_baseline",
]


def validate_required_columns(df: pd.DataFrame, required_cols: list[str], df_name: str) -> None:
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {df_name}: {missing}")


def build_simple_rank_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add easy-to-interpret derived rank features.
    Higher deprivation intensity proxies are created by inverting rank direction:
    lower official rank = more deprived, so we convert to descending severity-style fields.
    """
    out = df.copy()

    rank_cols = [
        "active_simd_rank_overall",
        "active_income_domain_rank",
        "active_employment_domain_rank",
        "active_health_domain_rank",
        "active_education_domain_rank",
        "active_housing_domain_rank",
        "active_access_domain_rank",
        "active_crime_domain_rank",
    ]

    for col in rank_cols:
        max_rank = out[col].max()
        out[f"{col}_severity"] = max_rank - out[col] + 1

    return out


def build_access_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a few simple ratio-style access features for modelling.
    """
    out = df.copy()

    eps = 1e-6

    out["bank_vs_any_ratio"] = out["dist_to_nearest_bank_m"] / (out["dist_to_nearest_any_access_point_m"] + eps)
    out["atm_vs_any_ratio"] = out["dist_to_nearest_atm_m"] / (out["dist_to_nearest_any_access_point_m"] + eps)
    out["post_vs_any_ratio"] = out["dist_to_nearest_post_office_m"] / (out["dist_to_nearest_any_access_point_m"] + eps)

    out["bank_minus_any_m"] = out["dist_to_nearest_bank_m"] - out["dist_to_nearest_any_access_point_m"]
    out["atm_minus_any_m"] = out["dist_to_nearest_atm_m"] - out["dist_to_nearest_any_access_point_m"]
    out["post_minus_any_m"] = out["dist_to_nearest_post_office_m"] - out["dist_to_nearest_any_access_point_m"]

    return out


def main() -> None:
    print("Starting ml_features_baseline_2022 build...")

    context_path = PROCESSED_DIR / "context" / "zone_year_context_2022.csv"
    access_path = PROCESSED_DIR / "accessibility" / "zone_accessibility_baseline_2022.csv"
    labels_path = PROCESSED_DIR / "accessibility" / "underserved_labels_baseline_2022.csv"

    ml_dir = PROCESSED_DIR / "ml"
    ml_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading context from: {context_path}")
    context_df = pd.read_csv(context_path)

    print(f"Loading accessibility from: {access_path}")
    access_df = pd.read_csv(access_path)

    print(f"Loading underserved labels from: {labels_path}")
    labels_df = pd.read_csv(labels_path)

    print(f"Context rows loaded: {len(context_df)}")
    print(f"Accessibility rows loaded: {len(access_df)}")
    print(f"Label rows loaded: {len(labels_df)}")

    validate_required_columns(context_df, CONTEXT_KEEP_COLS, "zone_year_context_2022")
    validate_required_columns(access_df, ACCESS_KEEP_COLS, "zone_accessibility_baseline_2022")
    validate_required_columns(labels_df, LABEL_KEEP_COLS, "underserved_labels_baseline_2022")

    # We want one row per 2022 Data Zone for the first baseline ML table.
    context_model = context_df[CONTEXT_KEEP_COLS].drop_duplicates(subset=["dz_code_2022"]).copy()
    access_model = access_df[ACCESS_KEEP_COLS].drop_duplicates(subset=["dz_code_2022"]).copy()
    labels_model = labels_df[LABEL_KEEP_COLS].drop_duplicates(subset=["dz_code_2022"]).copy()

    print("\nRows after zone-level deduplication:")
    print(f"Context model rows: {len(context_model)}")
    print(f"Accessibility model rows: {len(access_model)}")
    print(f"Labels model rows: {len(labels_model)}")

    ml_features = context_model.merge(
        access_model,
        on="dz_code_2022",
        how="left",
        validate="1:1",
    )

    ml_features = ml_features.merge(
        labels_model,
        on="dz_code_2022",
        how="left",
        validate="1:1",
    )

    print(f"\nRows after merging all inputs: {len(ml_features)}")

    ml_features = build_simple_rank_features(ml_features)
    ml_features = build_access_ratio_features(ml_features)

    missing_target = ml_features["underserved_baseline"].isna().sum()
    missing_critical_target = ml_features["critical_underserved_baseline"].isna().sum()

    print(f"Missing underserved_baseline values: {missing_target}")
    print(f"Missing critical_underserved_baseline values: {missing_critical_target}")

    print("\nTarget distribution:")
    print(ml_features["underserved_baseline"].value_counts(dropna=False))

    print("\nCritical target distribution:")
    print(ml_features["critical_underserved_baseline"].value_counts(dropna=False))

    print("\nRows by rural flag:")
    print(ml_features["is_rural"].value_counts(dropna=False))

    print("\nRows by UR6:")
    print(ml_features["ur6_name"].value_counts(dropna=False))

    output_csv = ml_dir / "ml_features_baseline_2022.csv"
    ml_features.to_csv(output_csv, index=False)

    print("\nPreview of ML feature table:")
    print(ml_features.head())

    print("\nml_features_baseline_2022 build completed successfully.")
    print(f"Output CSV: {output_csv}")


if __name__ == "__main__":
    main()