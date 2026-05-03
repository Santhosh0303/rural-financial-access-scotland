from pathlib import Path
import pandas as pd

from config.paths import PROCESSED_DIR


def fix_candidate_reason(row: pd.Series) -> str:
    """
    Repair the reason logic so every shortlisted intervention candidate has
    an intervention-style explanation.
    """
    if row["critical_underserved_baseline"] == 1 and row["ur6_name"] == "Remote Rural":
        return "remote_rural_critical_underserved"

    if row["critical_underserved_baseline"] == 1:
        return "critical_underserved_baseline"

    if row["underserved_baseline"] == 1 and row["ur6_name"] == "Remote Rural":
        return "remote_rural_high_risk"

    if row["underserved_baseline"] == 1 and row["ur6_name"] == "Accessible Rural":
        return "accessible_rural_high_gap"

    if row["preferred_model_predicted_class"] == 1:
        return "model_predicted_underserved"

    if row["preferred_risk_band"] == "high":
        return "high_risk_band_candidate"

    return "monitor_only"


def infer_primary_gap_service(row: pd.Series) -> str:
    """
    Identify which service type currently appears most difficult to access.
    """
    distance_map = {
        "bank": row["dist_to_nearest_bank_km"],
        "atm": row["dist_to_nearest_atm_km"],
        "post_office": row["dist_to_nearest_post_office_km"],
    }

    return max(distance_map, key=distance_map.get)


def assign_recommended_intervention(row: pd.Series) -> str:
    """
    Assign a first-pass intervention recommendation.
    This stays deliberately simple and policy-readable.
    """
    if row["critical_underserved_baseline"] == 1 and row["is_remote_rural"] == 1:
        return "multi_service_access_candidate"

    if row["primary_gap_service"] == "bank":
        return "new_bank_access_candidate"

    if row["primary_gap_service"] == "atm":
        return "new_atm_candidate"

    if row["primary_gap_service"] == "post_office":
        return "new_post_office_access_candidate"

    return "further_review_required"


def assign_intervention_tier(row: pd.Series) -> str:
    """
    Create a simple tier for prioritisation.
    """
    if row["critical_underserved_baseline"] == 1:
        return "tier_1_critical"

    if row["underserved_baseline"] == 1 or row["preferred_model_predicted_class"] == 1:
        return "tier_2_high_priority"

    if row["preferred_risk_band"] == "high":
        return "tier_3_watchlist"

    return "monitor_only"


def main() -> None:
    print("Starting scenario intervention build...")

    scenario_dir = PROCESSED_DIR / "scenario"
    shortlist_path = scenario_dir / "scenario_candidates_intervention_shortlist.csv"

    print(f"Loading intervention shortlist from: {shortlist_path}")
    shortlist = pd.read_csv(shortlist_path)

    print(f"Shortlist rows loaded: {len(shortlist)}")

    required_cols = [
        "dz_code_2022",
        "dz_name_2022",
        "ur6_name",
        "ur8_name",
        "is_accessible_rural",
        "is_remote_rural",
        "preferred_model_probability",
        "preferred_model_predicted_class",
        "preferred_risk_band",
        "underserved_baseline",
        "critical_underserved_baseline",
        "dist_to_nearest_bank_km",
        "dist_to_nearest_atm_km",
        "dist_to_nearest_post_office_km",
        "dist_to_nearest_any_access_point_km",
    ]

    missing = [col for col in required_cols if col not in shortlist.columns]
    if missing:
        raise ValueError(f"Missing required columns in shortlist: {missing}")

    interventions = shortlist.copy()

    interventions["candidate_reason_fixed"] = interventions.apply(fix_candidate_reason, axis=1)
    interventions["primary_gap_service"] = interventions.apply(infer_primary_gap_service, axis=1)
    interventions["recommended_intervention"] = interventions.apply(assign_recommended_intervention, axis=1)
    interventions["intervention_tier"] = interventions.apply(assign_intervention_tier, axis=1)

    interventions = interventions.sort_values(
        by=["scenario_priority_rank", "preferred_model_probability"],
        ascending=[True, False],
    ).reset_index(drop=True)

    top_100 = interventions.head(100).copy()
    top_20 = interventions.head(20).copy()

    print("\n--- Scenario Intervention Summary ---")
    print(f"Intervention design rows: {len(interventions)}")
    print(f"Top 100 rows: {len(top_100)}")
    print(f"Top 20 rows: {len(top_20)}")

    print("\nFixed candidate reason counts:")
    print(interventions["candidate_reason_fixed"].value_counts(dropna=False))

    print("\nPrimary gap service counts:")
    print(interventions["primary_gap_service"].value_counts(dropna=False))

    print("\nRecommended intervention counts:")
    print(interventions["recommended_intervention"].value_counts(dropna=False))

    print("\nIntervention tier counts:")
    print(interventions["intervention_tier"].value_counts(dropna=False))

    print("\nTop 10 intervention designs:")
    print(
        interventions[
            [
                "dz_code_2022",
                "dz_name_2022",
                "ur6_name",
                "ur8_name",
                "preferred_model_probability",
                "preferred_risk_band",
                "underserved_baseline",
                "critical_underserved_baseline",
                "primary_gap_service",
                "recommended_intervention",
                "candidate_reason_fixed",
                "intervention_tier",
            ]
        ].head(10)
    )

    all_csv = scenario_dir / "scenario_interventions_all.csv"
    top100_csv = scenario_dir / "scenario_interventions_top_100.csv"
    top20_csv = scenario_dir / "scenario_interventions_top_20.csv"

    interventions.to_csv(all_csv, index=False)
    top_100.to_csv(top100_csv, index=False)
    top_20.to_csv(top20_csv, index=False)

    print("\nScenario intervention build completed successfully.")
    print(f"All intervention designs CSV: {all_csv}")
    print(f"Top 100 intervention designs CSV: {top100_csv}")
    print(f"Top 20 intervention designs CSV: {top20_csv}")


if __name__ == "__main__":
    main()