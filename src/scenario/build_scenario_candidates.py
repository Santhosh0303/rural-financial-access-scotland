from pathlib import Path
import pandas as pd

from config.paths import PROCESSED_DIR


def assign_candidate_reason(row: pd.Series) -> str:
    """
    Assign a simple, explainable scenario-candidate reason.
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
    return "monitor_only"


def main() -> None:
    print("Starting scenario candidate build...")

    ml_dir = PROCESSED_DIR / "ml"
    prediction_path = ml_dir / "prediction_outputs" / "rural_zone_prediction_outputs.csv"
    master_features_path = ml_dir / "ml_features_baseline_2022.csv"

    scenario_dir = PROCESSED_DIR / "scenario"
    scenario_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading rural prediction outputs from: {prediction_path}")
    predictions = pd.read_csv(prediction_path)

    print(f"Loading ML master features from: {master_features_path}")
    master_features = pd.read_csv(master_features_path)

    print(f"Prediction rows loaded: {len(predictions)}")
    print(f"Master feature rows loaded: {len(master_features)}")

    # Bring back the current accessibility distances for scenario interpretation
    distance_cols = [
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

    distance_cols = [col for col in distance_cols if col in master_features.columns]
    distance_lookup = master_features[distance_cols].drop_duplicates(subset=["dz_code_2022"]).copy()

    scenario_candidates = predictions.merge(
        distance_lookup,
        on="dz_code_2022",
        how="left",
        validate="1:1",
    )

    scenario_candidates["candidate_reason"] = scenario_candidates.apply(assign_candidate_reason, axis=1)

    # Priority score: high probability + stronger baseline problem indicators
    scenario_candidates["scenario_priority_score"] = (
        scenario_candidates["preferred_model_probability"] * 100
        + scenario_candidates["critical_underserved_baseline"] * 20
        + scenario_candidates["underserved_baseline"] * 10
        + scenario_candidates["preferred_model_predicted_class"] * 10
    )

    scenario_candidates = scenario_candidates.sort_values(
        by=["scenario_priority_score", "preferred_model_probability"],
        ascending=False,
    ).reset_index(drop=True)

    scenario_candidates["scenario_priority_rank"] = scenario_candidates.index + 1

    # Shortlists
    intervention_candidates = scenario_candidates[
        (
            (scenario_candidates["critical_underserved_baseline"] == 1)
            | (scenario_candidates["preferred_model_predicted_class"] == 1)
            | (scenario_candidates["preferred_risk_band"] == "high")
        )
    ].copy()

    top_100_candidates = intervention_candidates.head(100).copy()
    top_20_candidates = intervention_candidates.head(20).copy()

    print("\n--- Scenario Candidate Summary ---")
    print(f"All rural scenario rows: {len(scenario_candidates)}")
    print(f"Intervention candidate rows: {len(intervention_candidates)}")
    print(f"Top 100 candidate rows: {len(top_100_candidates)}")
    print(f"Top 20 candidate rows: {len(top_20_candidates)}")

    print("\nCandidate reason counts:")
    print(intervention_candidates["candidate_reason"].value_counts(dropna=False))

    print("\nTop 10 intervention candidates:")
    print(
        intervention_candidates[
            [
                "dz_code_2022",
                "dz_name_2022",
                "ur6_name",
                "ur8_name",
                "preferred_model_probability",
                "preferred_model_predicted_class",
                "preferred_risk_band",
                "underserved_baseline",
                "critical_underserved_baseline",
                "dist_to_nearest_any_access_point_km",
                "candidate_reason",
                "scenario_priority_rank",
            ]
        ].head(10)
    )

    all_candidates_csv = scenario_dir / "scenario_candidates_all_rural.csv"
    intervention_candidates_csv = scenario_dir / "scenario_candidates_intervention_shortlist.csv"
    top_100_csv = scenario_dir / "scenario_candidates_top_100.csv"
    top_20_csv = scenario_dir / "scenario_candidates_top_20.csv"

    scenario_candidates.to_csv(all_candidates_csv, index=False)
    intervention_candidates.to_csv(intervention_candidates_csv, index=False)
    top_100_candidates.to_csv(top_100_csv, index=False)
    top_20_candidates.to_csv(top_20_csv, index=False)

    print("\nScenario candidate build completed successfully.")
    print(f"All rural candidates CSV: {all_candidates_csv}")
    print(f"Intervention shortlist CSV: {intervention_candidates_csv}")
    print(f"Top 100 candidates CSV: {top_100_csv}")
    print(f"Top 20 candidates CSV: {top_20_csv}")


if __name__ == "__main__":
    main()