from pathlib import Path
import pandas as pd

from config.paths import PROCESSED_DIR


DISTANCE_COLS = [
    "dist_to_nearest_bank_m",
    "dist_to_nearest_atm_m",
    "dist_to_nearest_post_office_m",
    "dist_to_nearest_any_access_point_m",
]


def build_group_summary(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """
    Build a wide summary table with count, mean, median, and p75 for each distance metric.
    """
    rows = []

    for group_value, group_df in df.groupby(group_col, dropna=False):
        row = {
            group_col: group_value,
            "zone_count": len(group_df),
        }

        for col in DISTANCE_COLS:
            row[f"{col}_mean_m"] = group_df[col].mean()
            row[f"{col}_median_m"] = group_df[col].median()
            row[f"{col}_p75_m"] = group_df[col].quantile(0.75)

        rows.append(row)

    summary_df = pd.DataFrame(rows)
    return summary_df


def build_rural_thresholds(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the rural threshold table used for the first underserved rule.
    """
    rural_df = df[df["is_rural"] == 1].copy()

    if rural_df.empty:
        raise ValueError("No rural zones found while building rural thresholds.")

    threshold_row = {
        "threshold_basis": "rural_p75",
        "zone_count_used": len(rural_df),
        "dist_to_nearest_bank_m_threshold": rural_df["dist_to_nearest_bank_m"].quantile(0.75),
        "dist_to_nearest_atm_m_threshold": rural_df["dist_to_nearest_atm_m"].quantile(0.75),
        "dist_to_nearest_post_office_m_threshold": rural_df["dist_to_nearest_post_office_m"].quantile(0.75),
        "dist_to_nearest_any_access_point_m_threshold": rural_df["dist_to_nearest_any_access_point_m"].quantile(0.75),
    }

    return pd.DataFrame([threshold_row])


def build_underserved_labels(df: pd.DataFrame, thresholds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the first rule-based underserved labels.
    """
    out = df.copy()

    threshold_row = thresholds_df.iloc[0]

    bank_threshold = threshold_row["dist_to_nearest_bank_m_threshold"]
    atm_threshold = threshold_row["dist_to_nearest_atm_m_threshold"]
    post_threshold = threshold_row["dist_to_nearest_post_office_m_threshold"]
    any_threshold = threshold_row["dist_to_nearest_any_access_point_m_threshold"]

    out["flag_far_bank"] = (out["dist_to_nearest_bank_m"] >= bank_threshold).astype(int)
    out["flag_far_atm"] = (out["dist_to_nearest_atm_m"] >= atm_threshold).astype(int)
    out["flag_far_post_office"] = (out["dist_to_nearest_post_office_m"] >= post_threshold).astype(int)
    out["flag_far_any_access_point"] = (
        out["dist_to_nearest_any_access_point_m"] >= any_threshold
    ).astype(int)

    out["service_gap_count"] = (
        out["flag_far_bank"] + out["flag_far_atm"] + out["flag_far_post_office"]
    )

    out["total_gap_score"] = out["service_gap_count"] + out["flag_far_any_access_point"]

    out["underserved_baseline"] = (
        (out["is_rural"] == 1)
        & (out["flag_far_any_access_point"] == 1)
        & (out["service_gap_count"] >= 1)
    ).astype(int)

    out["critical_underserved_baseline"] = (
        (out["is_rural"] == 1)
        & (out["flag_far_any_access_point"] == 1)
        & (out["service_gap_count"] >= 2)
    ).astype(int)

    # Dominant service gap for interpretation
    dominant_map = {
        "dist_to_nearest_bank_m": "bank",
        "dist_to_nearest_atm_m": "atm",
        "dist_to_nearest_post_office_m": "post_office",
    }

    out["dominant_gap_type"] = (
        out[
            [
                "dist_to_nearest_bank_m",
                "dist_to_nearest_atm_m",
                "dist_to_nearest_post_office_m",
            ]
        ]
        .idxmax(axis=1)
        .map(dominant_map)
    )

    keep_cols = [
        "dz_code_2022",
        "dz_name_2022",
        "ur6_name",
        "ur8_name",
        "is_rural",
        "is_accessible_rural",
        "is_remote_rural",
        "dist_to_nearest_bank_m",
        "dist_to_nearest_atm_m",
        "dist_to_nearest_post_office_m",
        "dist_to_nearest_any_access_point_m",
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

    out = out[keep_cols].copy()
    return out


def main() -> None:
    print("Starting rural accessibility summary and underserved baseline build...")

    accessibility_dir = PROCESSED_DIR / "accessibility"

    zone_access_path = accessibility_dir / "zone_accessibility_baseline_2022.csv"
    zone_year_access_path = accessibility_dir / "zone_year_accessibility_baseline_2022.csv"

    print(f"Loading zone accessibility from: {zone_access_path}")
    zone_access = pd.read_csv(zone_access_path)

    print(f"Loading zone-year accessibility from: {zone_year_access_path}")
    zone_year_access = pd.read_csv(zone_year_access_path)

    print(f"Zone accessibility rows: {len(zone_access)}")
    print(f"Zone-year accessibility rows: {len(zone_year_access)}")

    # Build summary tables
    print("\nBuilding rural vs non-rural summary...")
    rural_nonrural_summary = build_group_summary(zone_access, "is_rural")

    print("Building UR6 summary...")
    ur6_summary = build_group_summary(zone_access, "ur6_name")

    print("Building rural threshold table...")
    thresholds_df = build_rural_thresholds(zone_access)

    print("Building first underserved labels...")
    underserved_labels = build_underserved_labels(zone_access, thresholds_df)

    print("Merging labels into the zone-year accessibility table...")
    zone_year_labels = zone_year_access.merge(
        underserved_labels[
            [
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
        ],
        on="dz_code_2022",
        how="left",
        validate="m:1",
    )

    # Save outputs
    rural_nonrural_summary_path = accessibility_dir / "rural_nonrural_accessibility_summary_2022.csv"
    ur6_summary_path = accessibility_dir / "ur6_accessibility_summary_2022.csv"
    thresholds_path = accessibility_dir / "underserved_thresholds_baseline_2022.csv"
    underserved_labels_path = accessibility_dir / "underserved_labels_baseline_2022.csv"
    zone_year_labels_path = accessibility_dir / "zone_year_underserved_baseline_2022.csv"

    rural_nonrural_summary.to_csv(rural_nonrural_summary_path, index=False)
    ur6_summary.to_csv(ur6_summary_path, index=False)
    thresholds_df.to_csv(thresholds_path, index=False)
    underserved_labels.to_csv(underserved_labels_path, index=False)
    zone_year_labels.to_csv(zone_year_labels_path, index=False)

    print("\n--- Rural Thresholds Used ---")
    print(thresholds_df)

    print("\nUnderserved baseline counts:")
    print(underserved_labels["underserved_baseline"].value_counts(dropna=False))

    print("\nCritical underserved baseline counts:")
    print(underserved_labels["critical_underserved_baseline"].value_counts(dropna=False))

    print("\nUnderserved counts by rural flag:")
    print(
        underserved_labels.groupby("is_rural")[
            ["underserved_baseline", "critical_underserved_baseline"]
        ].sum()
    )

    print("\nTop UR6 summary preview:")
    print(ur6_summary.head())

    print("\nRural accessibility summary and underserved baseline build completed successfully.")
    print(f"Rural/non-rural summary: {rural_nonrural_summary_path}")
    print(f"UR6 summary: {ur6_summary_path}")
    print(f"Thresholds: {thresholds_path}")
    print(f"Underserved labels: {underserved_labels_path}")
    print(f"Zone-year underserved labels: {zone_year_labels_path}")


if __name__ == "__main__":
    main()