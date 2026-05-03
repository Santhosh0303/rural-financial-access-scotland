import numpy as np
import pandas as pd

from config.paths import PROCESSED_DIR


PRE_COVID_YEARS = [2015, 2016, 2017, 2018, 2019]
COVID_YEAR = [2020]
POST_COVID_YEARS = [2021, 2022, 2023]


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return np.where(denominator > 0, numerator / denominator, np.nan)


def build_period_agg(panel: pd.DataFrame, years: list[int], prefix: str) -> pd.DataFrame:
    subset = panel[panel["year"].isin(years)].copy()

    grouped = (
        subset.groupby("dz_code_2022", dropna=False)
        .agg(
            **{
                f"{prefix}_closures_total": ("closures_in_year", "sum"),
                f"{prefix}_years_with_closure": ("closure_event_flag", "sum"),
                f"{prefix}_unique_brands_total": ("unique_closing_brands", "sum"),
                f"{prefix}_unique_branch_types_total": ("unique_closing_branch_types", "sum"),
            }
        )
        .reset_index()
    )

    grouped[f"{prefix}_year_count"] = len(years)
    grouped[f"{prefix}_closures_annual_rate"] = (
        grouped[f"{prefix}_closures_total"] / grouped[f"{prefix}_year_count"]
    )

    return grouped


def main() -> None:
    print("Starting bank closure change feature build...")

    processed_dir = PROCESSED_DIR / "bank_closures"
    panel_path = processed_dir / "zone_year_bank_closure_panel_2022.csv"

    print(f"Loading zone-year closure panel from: {panel_path}")
    panel = pd.read_csv(panel_path)

    print(f"Rows loaded: {len(panel)}")
    print(f"Columns loaded: {len(panel.columns)}")

    required_cols = [
        "dz_code_2022",
        "dz_name_2022",
        "year",
        "ur6_name",
        "ur8_name",
        "is_rural",
        "is_accessible_rural",
        "is_remote_rural",
        "closures_in_year",
        "closure_event_flag",
        "unique_closing_brands",
        "unique_closing_branch_types",
        "first_closure_year",
        "cumulative_closures_to_year",
    ]
    missing = [col for col in required_cols if col not in panel.columns]
    if missing:
        raise ValueError(f"Missing required columns in zone-year closure panel: {missing}")

    zone_lookup = (
        panel[
            [
                "dz_code_2022",
                "dz_name_2022",
                "ur6_name",
                "ur8_name",
                "is_rural",
                "is_accessible_rural",
                "is_remote_rural",
            ]
        ]
        .drop_duplicates(subset=["dz_code_2022"])
        .copy()
    )

    pre_agg = build_period_agg(panel, PRE_COVID_YEARS, "pre_covid")
    covid_agg = build_period_agg(panel, COVID_YEAR, "covid_transition")
    post_agg = build_period_agg(panel, POST_COVID_YEARS, "post_covid")

    zone_change = zone_lookup.merge(pre_agg, on="dz_code_2022", how="left", validate="1:1")
    zone_change = zone_change.merge(covid_agg, on="dz_code_2022", how="left", validate="1:1")
    zone_change = zone_change.merge(post_agg, on="dz_code_2022", how="left", validate="1:1")

    fill_zero_cols = [col for col in zone_change.columns if any(
        token in col for token in [
            "_closures_total",
            "_years_with_closure",
            "_unique_brands_total",
            "_unique_branch_types_total",
            "_year_count",
        ]
    )]
    for col in fill_zero_cols:
        zone_change[col] = zone_change[col].fillna(0)

    annual_rate_cols = [col for col in zone_change.columns if col.endswith("_closures_annual_rate")]
    for col in annual_rate_cols:
        zone_change[col] = zone_change[col].fillna(0.0)

    first_closure_lookup = (
        panel.groupby("dz_code_2022", dropna=False)["first_closure_year"]
        .min()
        .reset_index()
    )

    cumulative_2023_lookup = (
        panel[panel["year"] == 2023][["dz_code_2022", "cumulative_closures_to_year"]]
        .rename(columns={"cumulative_closures_to_year": "cumulative_closures_to_2023"})
        .copy()
    )

    zone_change = zone_change.merge(
        first_closure_lookup,
        on="dz_code_2022",
        how="left",
        validate="1:1",
    )
    zone_change = zone_change.merge(
        cumulative_2023_lookup,
        on="dz_code_2022",
        how="left",
        validate="1:1",
    )

    zone_change["any_pre_covid_closure_flag"] = (zone_change["pre_covid_closures_total"] > 0).astype(int)
    zone_change["any_post_covid_closure_flag"] = (zone_change["post_covid_closures_total"] > 0).astype(int)
    zone_change["post_covid_only_closure_flag"] = (
        (zone_change["pre_covid_closures_total"] == 0)
        & (zone_change["post_covid_closures_total"] > 0)
    ).astype(int)

    zone_change["closure_rate_change_post_minus_pre"] = (
        zone_change["post_covid_closures_annual_rate"]
        - zone_change["pre_covid_closures_annual_rate"]
    )

    zone_change["closure_rate_ratio_post_to_pre"] = safe_ratio(
        zone_change["post_covid_closures_annual_rate"],
        zone_change["pre_covid_closures_annual_rate"],
    )

    zone_change["closure_count_change_post_minus_pre"] = (
        zone_change["post_covid_closures_total"]
        - zone_change["pre_covid_closures_total"]
    )

    zone_change["closure_deterioration_flag"] = (
        zone_change["closure_rate_change_post_minus_pre"] > 0
    ).astype(int)

    zone_change["closure_persistence_flag"] = (
        (zone_change["any_pre_covid_closure_flag"] == 1)
        & (zone_change["any_post_covid_closure_flag"] == 1)
    ).astype(int)

    zone_change["closure_change_rank"] = (
        zone_change.sort_values(
            by=[
                "closure_rate_change_post_minus_pre",
                "post_covid_closures_total",
                "cumulative_closures_to_2023",
            ],
            ascending=[False, False, False],
        )
        .reset_index(drop=True)
        .index
        + 1
    )

    zone_change = zone_change.sort_values(
        by=[
            "closure_rate_change_post_minus_pre",
            "post_covid_closures_total",
            "cumulative_closures_to_2023",
        ],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    zone_change["closure_change_rank"] = zone_change.index + 1

    rural_summary = (
        zone_change[zone_change["is_rural"] == 1]
        .groupby("ur6_name", dropna=False)
        .agg(
            rural_zone_count=("dz_code_2022", "count"),
            pre_covid_closures_total=("pre_covid_closures_total", "sum"),
            post_covid_closures_total=("post_covid_closures_total", "sum"),
            mean_pre_covid_annual_rate=("pre_covid_closures_annual_rate", "mean"),
            mean_post_covid_annual_rate=("post_covid_closures_annual_rate", "mean"),
            mean_rate_change=("closure_rate_change_post_minus_pre", "mean"),
            deteriorating_zone_count=("closure_deterioration_flag", "sum"),
            post_covid_only_zone_count=("post_covid_only_closure_flag", "sum"),
            persistent_closure_zone_count=("closure_persistence_flag", "sum"),
        )
        .reset_index()
        .sort_values("mean_rate_change", ascending=False)
        .reset_index(drop=True)
    )

    overall_summary = pd.DataFrame(
        {
            "metric": [
                "total_zones",
                "rural_zones",
                "zones_with_any_pre_covid_closure",
                "zones_with_any_post_covid_closure",
                "zones_with_post_covid_only_closure",
                "zones_with_closure_deterioration",
                "total_pre_covid_closures",
                "total_post_covid_closures",
            ],
            "value": [
                int(len(zone_change)),
                int((zone_change["is_rural"] == 1).sum()),
                int(zone_change["any_pre_covid_closure_flag"].sum()),
                int(zone_change["any_post_covid_closure_flag"].sum()),
                int(zone_change["post_covid_only_closure_flag"].sum()),
                int(zone_change["closure_deterioration_flag"].sum()),
                int(zone_change["pre_covid_closures_total"].sum()),
                int(zone_change["post_covid_closures_total"].sum()),
            ],
        }
    )

    top_100_worsening_rural = (
        zone_change[zone_change["is_rural"] == 1]
        .copy()
        .sort_values(
            by=[
                "closure_rate_change_post_minus_pre",
                "post_covid_closures_total",
                "cumulative_closures_to_2023",
            ],
            ascending=[False, False, False],
        )
        .reset_index(drop=True)
    )
    top_100_worsening_rural["rural_worsening_rank"] = top_100_worsening_rural.index + 1

    top_100_worsening_rural = top_100_worsening_rural[
        [
            "rural_worsening_rank",
            "dz_code_2022",
            "dz_name_2022",
            "ur6_name",
            "ur8_name",
            "pre_covid_closures_total",
            "post_covid_closures_total",
            "pre_covid_closures_annual_rate",
            "post_covid_closures_annual_rate",
            "closure_rate_change_post_minus_pre",
            "closure_count_change_post_minus_pre",
            "post_covid_only_closure_flag",
            "closure_persistence_flag",
            "first_closure_year",
            "cumulative_closures_to_2023",
        ]
    ].head(100)

    print("\n--- Bank Closure Change Feature Summary ---")
    print(f"Zone change rows: {len(zone_change)}")
    print(f"Rural summary rows: {len(rural_summary)}")
    print(f"Top 100 worsening rural rows: {len(top_100_worsening_rural)}")

    print("\nOverall summary:")
    print(overall_summary)

    print("\nRural summary preview:")
    print(rural_summary.head(10))

    print("\nTop 10 worsening rural zones:")
    print(top_100_worsening_rural.head(10))

    zone_change_path = processed_dir / "bank_closure_change_features_2022.csv"
    rural_summary_path = processed_dir / "bank_closure_change_rural_summary_2022.csv"
    overall_summary_path = processed_dir / "bank_closure_change_overall_summary_2022.csv"
    top100_path = processed_dir / "bank_closure_top_100_worsening_rural_zones_2022.csv"

    zone_change.to_csv(zone_change_path, index=False)
    rural_summary.to_csv(rural_summary_path, index=False)
    overall_summary.to_csv(overall_summary_path, index=False)
    top_100_worsening_rural.to_csv(top100_path, index=False)

    print("\nBank closure change feature build completed successfully.")
    print(f"Zone change feature CSV: {zone_change_path}")
    print(f"Rural summary CSV: {rural_summary_path}")
    print(f"Overall summary CSV: {overall_summary_path}")
    print(f"Top 100 worsening rural CSV: {top100_path}")


if __name__ == "__main__":
    main()