"""
Final QA and Evidence Report Builder
MSc Dissertation: Rural Financial Access in Scotland

Purpose:
- Check that final processed datasets, ML outputs, scenario outputs, and dashboard exports exist.
- Validate row counts, required columns, coordinate ranges, distance values, model metrics, and scenario outputs.
- Produce evidence files for README, pipeline documentation, data validation, model evaluation,
  scenario-method write-up, and final dissertation evidence.

Important note:
- Model probabilities and shares are checked on a 0–1 scale.
- Scenario percentage-improvement fields are checked on a 0–100 percentage scale.

Run from project root:
    python -m src.qa.run_final_project_qa
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import geopandas as gpd
except ImportError:
    # This keeps the script readable if geopandas is not available,
    # but the geospatial checks will fail clearly.
    gpd = None


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path.cwd()
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
QA_DIR = PROCESSED_DIR / "qa"


# ---------------------------------------------------------------------
# Broad Scotland coordinate bounds for final map-ready QA checks.
# These are intentionally broad so valid Scotland locations are not rejected.
# ---------------------------------------------------------------------

SCOTLAND_LAT_MIN = 54.0
SCOTLAND_LAT_MAX = 61.5
SCOTLAND_LON_MIN = -9.5
SCOTLAND_LON_MAX = -0.5


# ---------------------------------------------------------------------
# QA result structure
# ---------------------------------------------------------------------

@dataclass
class QAResult:
    check_name: str
    status: str
    file_path: str
    message: str
    rows: int | None = None
    columns: int | None = None
    details: dict[str, Any] | list[Any] | None = None


results: list[QAResult] = []


def add_result(
    check_name: str,
    status: str,
    file_path: Path | str,
    message: str,
    rows: int | None = None,
    columns: int | None = None,
    details: dict[str, Any] | list[Any] | None = None,
) -> None:
    """Add one QA result record."""
    results.append(
        QAResult(
            check_name=check_name,
            status=status,
            file_path=str(file_path),
            message=message,
            rows=rows,
            columns=columns,
            details=details or {},
        )
    )


# ---------------------------------------------------------------------
# Safe readers
# ---------------------------------------------------------------------

def read_csv_safe(path: Path, check_name: str, required: bool = True) -> pd.DataFrame | None:
    """Safely load a CSV file and record PASS/FAIL/WARN."""
    if not path.exists():
        status = "FAIL" if required else "WARN"
        add_result(check_name, status, path, "File not found.")
        return None

    try:
        df = pd.read_csv(path)
        add_result(
            check_name,
            "PASS",
            path,
            "File loaded successfully.",
            rows=len(df),
            columns=len(df.columns),
            details={"columns": list(df.columns)},
        )
        return df
    except Exception as exc:
        add_result(check_name, "FAIL", path, f"Failed to load CSV: {exc}")
        return None


def read_json_safe(path: Path, check_name: str, required: bool = True) -> dict[str, Any] | None:
    """Safely load a JSON file and record PASS/FAIL/WARN."""
    if not path.exists():
        status = "FAIL" if required else "WARN"
        add_result(check_name, status, path, "JSON file not found.")
        return None

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        add_result(check_name, "PASS", path, "JSON loaded successfully.", details=data)
        return data
    except Exception as exc:
        add_result(check_name, "FAIL", path, f"Failed to load JSON: {exc}")
        return None


def read_gpkg_safe(path: Path, check_name: str, required: bool = True):
    """Safely load a GeoPackage file and record PASS/FAIL/WARN."""
    if gpd is None:
        add_result(check_name, "FAIL", path, "geopandas is not installed/importable.")
        return None

    if not path.exists():
        status = "FAIL" if required else "WARN"
        add_result(check_name, status, path, "GeoPackage file not found.")
        return None

    try:
        gdf = gpd.read_file(path)
        add_result(
            check_name,
            "PASS",
            path,
            "GeoPackage loaded successfully.",
            rows=len(gdf),
            columns=len(gdf.columns),
            details={"crs": str(gdf.crs), "columns": list(gdf.columns)},
        )
        return gdf
    except Exception as exc:
        add_result(check_name, "FAIL", path, f"Failed to load GeoPackage: {exc}")
        return None


# ---------------------------------------------------------------------
# Generic QA helpers
# ---------------------------------------------------------------------

def require_columns(df: pd.DataFrame, cols: list[str], file_path: Path, check_name: str) -> None:
    """Check whether all required columns exist."""
    missing = [col for col in cols if col not in df.columns]

    if missing:
        add_result(
            check_name,
            "FAIL",
            file_path,
            f"Missing required columns: {missing}",
            rows=len(df),
            columns=len(df.columns),
            details={"missing_columns": missing},
        )
    else:
        add_result(
            check_name,
            "PASS",
            file_path,
            "All required columns are present.",
            rows=len(df),
            columns=len(df.columns),
            details={"required_columns": cols},
        )


def check_row_count(
    df: pd.DataFrame,
    expected: int,
    file_path: Path,
    check_name: str,
    tolerance: int = 0,
) -> None:
    """Check row count against expected value, with optional tolerance."""
    actual = len(df)

    if abs(actual - expected) <= tolerance:
        add_result(
            check_name,
            "PASS",
            file_path,
            f"Row count matches expected value: {actual}.",
            rows=actual,
            columns=len(df.columns),
            details={"expected": expected, "actual": actual, "tolerance": tolerance},
        )
    else:
        add_result(
            check_name,
            "FAIL",
            file_path,
            f"Unexpected row count. Expected {expected}, got {actual}.",
            rows=actual,
            columns=len(df.columns),
            details={"expected": expected, "actual": actual, "tolerance": tolerance},
        )


def check_unique_key(df: pd.DataFrame, key: str, file_path: Path, check_name: str) -> None:
    """Check whether a key column is unique."""
    if key not in df.columns:
        add_result(check_name, "WARN", file_path, f"Cannot check uniqueness; missing key: {key}")
        return

    duplicate_count = int(df[key].duplicated().sum())

    if duplicate_count == 0:
        add_result(
            check_name,
            "PASS",
            file_path,
            f"{key} is unique.",
            rows=len(df),
            columns=len(df.columns),
        )
    else:
        add_result(
            check_name,
            "FAIL",
            file_path,
            f"{key} has {duplicate_count} duplicate rows.",
            rows=len(df),
            columns=len(df.columns),
            details={"duplicate_count": duplicate_count},
        )


def check_non_negative_columns(df: pd.DataFrame, cols: list[str], file_path: Path, check_name: str) -> None:
    """
    Check that numeric columns do not contain negative values.
    Missing columns are ignored here because required-column checks handle that separately.
    """
    failing: dict[str, int] = {}

    for col in cols:
        if col in df.columns:
            numeric = pd.to_numeric(df[col], errors="coerce")
            negative_count = int((numeric < 0).sum())
            if negative_count > 0:
                failing[col] = negative_count

    if failing:
        add_result(
            check_name,
            "FAIL",
            file_path,
            f"Negative values found in columns: {failing}",
            rows=len(df),
            columns=len(df.columns),
            details=failing,
        )
    else:
        add_result(
            check_name,
            "PASS",
            file_path,
            "No negative values found in checked numeric distance/count columns.",
            rows=len(df),
            columns=len(df.columns),
            details={"checked_columns": cols},
        )


def check_coordinate_range(df: pd.DataFrame, file_path: Path, check_name: str) -> None:
    """Check latitude/longitude values are within broad Scotland bounds."""
    required = ["latitude", "longitude"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        add_result(check_name, "WARN", file_path, f"Cannot check coordinates; missing {missing}.")
        return

    lat = pd.to_numeric(df["latitude"], errors="coerce")
    lon = pd.to_numeric(df["longitude"], errors="coerce")

    invalid = df[
        lat.isna()
        | lon.isna()
        | (lat < SCOTLAND_LAT_MIN)
        | (lat > SCOTLAND_LAT_MAX)
        | (lon < SCOTLAND_LON_MIN)
        | (lon > SCOTLAND_LON_MAX)
    ]

    if invalid.empty:
        add_result(
            check_name,
            "PASS",
            file_path,
            "Latitude/longitude values fall within broad Scotland bounds.",
            rows=len(df),
            columns=len(df.columns),
            details={
                "lat_min": float(lat.min()),
                "lat_max": float(lat.max()),
                "lon_min": float(lon.min()),
                "lon_max": float(lon.max()),
            },
        )
    else:
        add_result(
            check_name,
            "FAIL",
            file_path,
            f"{len(invalid)} rows have missing/out-of-range coordinates.",
            rows=len(df),
            columns=len(df.columns),
            details={"invalid_coordinate_rows": int(len(invalid))},
        )


def check_metric_range_0_1(df: pd.DataFrame, cols: list[str], file_path: Path, check_name: str) -> None:
    """
    Check probability/share/ratio fields are within 0–1.
    Missing columns are ignored here because required-column checks handle that separately.
    """
    issues: dict[str, int] = {}

    for col in cols:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce")
            issue_count = int(values.isna().sum() + ((values < 0) | (values > 1)).sum())
            if issue_count > 0:
                issues[col] = issue_count

    if issues:
        add_result(
            check_name,
            "FAIL",
            file_path,
            f"0–1 range issues found: {issues}",
            rows=len(df),
            columns=len(df.columns),
            details=issues,
        )
    else:
        add_result(
            check_name,
            "PASS",
            file_path,
            "Checked percentage/probability/share fields are within 0–1 range.",
            rows=len(df),
            columns=len(df.columns),
            details={"checked_columns": cols},
        )


def check_percentage_range_0_100(
    df: pd.DataFrame,
    cols: list[str],
    file_path: Path,
    check_name: str,
) -> None:
    """
    Check percentage fields stored on a 0–100 scale.

    This is used for scenario percentage-improvement columns such as:
    - bank_pct_improvement
    - atm_pct_improvement
    - post_office_pct_improvement
    - any_access_point_pct_improvement

    These are percentage values, not 0–1 probability/share ratios.
    """
    issues: dict[str, int] = {}

    for col in cols:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce")
            issue_count = int(values.isna().sum() + ((values < 0) | (values > 100)).sum())

            if issue_count > 0:
                issues[col] = issue_count

    if issues:
        add_result(
            check_name,
            "FAIL",
            file_path,
            f"0–100 percentage range issues found: {issues}",
            rows=len(df),
            columns=len(df.columns),
            details=issues,
        )
    else:
        add_result(
            check_name,
            "PASS",
            file_path,
            "Checked percentage-improvement fields are within 0–100 range.",
            rows=len(df),
            columns=len(df.columns),
            details={"checked_columns": cols},
        )


def summarise_dataframe_numeric(df: pd.DataFrame, numeric_cols: list[str]) -> dict[str, Any]:
    """Create simple numeric summary statistics for evidence reporting."""
    summary: dict[str, Any] = {}

    for col in numeric_cols:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce")
            summary[col] = {
                "non_null": int(values.notna().sum()),
                "mean": float(values.mean()) if values.notna().any() else None,
                "median": float(values.median()) if values.notna().any() else None,
                "min": float(values.min()) if values.notna().any() else None,
                "max": float(values.max()) if values.notna().any() else None,
            }

    return summary


# ---------------------------------------------------------------------
# Core geography QA
# ---------------------------------------------------------------------

def qa_core_geography() -> None:
    """Validate the master 2022 Data Zone geography."""
    zones_path = PROCESSED_DIR / "geography" / "zones_master_2022.gpkg"
    zones = read_gpkg_safe(zones_path, "core_geography_zones_master")

    if zones is not None:
        check_row_count(zones, 7392, zones_path, "core_geography_expected_7392_zones")

        if "dz_code_2022" in zones.columns:
            zones_no_geometry = pd.DataFrame(zones.drop(columns="geometry", errors="ignore"))
            check_unique_key(
                zones_no_geometry,
                "dz_code_2022",
                zones_path,
                "core_geography_unique_dz_code",
            )


# ---------------------------------------------------------------------
# Context and accessibility QA
# ---------------------------------------------------------------------

def qa_context_and_accessibility() -> None:
    """Validate context panel and accessibility baseline outputs."""
    context_path = PROCESSED_DIR / "context" / "zone_year_context_2022.csv"
    context = read_csv_safe(context_path, "context_zone_year_context")

    if context is not None:
        require_columns(
            context,
            ["dz_code_2022", "year", "population_total", "older_population_65_plus"],
            context_path,
            "context_required_columns",
        )

        check_non_negative_columns(
            context,
            ["population_total", "older_population_65_plus"],
            context_path,
            "context_population_non_negative",
        )

    baseline_path = PROCESSED_DIR / "accessibility" / "zone_accessibility_baseline_2022.csv"
    baseline = read_csv_safe(baseline_path, "accessibility_baseline_2022")

    if baseline is not None:
        require_columns(
            baseline,
            [
                "dz_code_2022",
                "dist_to_nearest_bank_km",
                "dist_to_nearest_atm_km",
                "dist_to_nearest_post_office_km",
                "dist_to_nearest_any_access_point_km",
            ],
            baseline_path,
            "accessibility_baseline_required_columns",
        )

        check_row_count(baseline, 7392, baseline_path, "accessibility_baseline_expected_7392_rows")
        check_unique_key(baseline, "dz_code_2022", baseline_path, "accessibility_baseline_unique_dz_code")

        check_non_negative_columns(
            baseline,
            [
                "dist_to_nearest_bank_km",
                "dist_to_nearest_atm_km",
                "dist_to_nearest_post_office_km",
                "dist_to_nearest_any_access_point_km",
            ],
            baseline_path,
            "accessibility_baseline_non_negative_distances",
        )

        add_result(
            "accessibility_baseline_numeric_summary",
            "INFO",
            baseline_path,
            "Numeric summary for accessibility baseline.",
            rows=len(baseline),
            columns=len(baseline.columns),
            details=summarise_dataframe_numeric(
                baseline,
                [
                    "dist_to_nearest_bank_km",
                    "dist_to_nearest_atm_km",
                    "dist_to_nearest_post_office_km",
                    "dist_to_nearest_any_access_point_km",
                ],
            ),
        )

    temporal_path = PROCESSED_DIR / "accessibility" / "bank_accessibility_temporal_summary_by_ur6.csv"
    temporal = read_csv_safe(temporal_path, "temporal_bank_accessibility_by_ur6", required=False)

    if temporal is not None:
        require_columns(
            temporal,
            ["ur6_name", "mean_bank_km_2019", "mean_bank_km_2023", "mean_bank_change_km"],
            temporal_path,
            "temporal_bank_required_columns",
        )

        check_non_negative_columns(
            temporal,
            ["mean_bank_km_2019", "mean_bank_km_2023"],
            temporal_path,
            "temporal_bank_non_negative_distances",
        )


# ---------------------------------------------------------------------
# ML QA
# ---------------------------------------------------------------------

def qa_ml_outputs() -> None:
    """Validate prediction outputs, CV metrics, and model evaluation evidence."""
    prediction_path = PROCESSED_DIR / "ml" / "prediction_outputs" / "rural_zone_prediction_outputs.csv"
    predictions = read_csv_safe(prediction_path, "ml_rural_prediction_outputs")

    if predictions is not None:
        require_columns(
            predictions,
            [
                "dz_code_2022",
                "preferred_model_probability",
                "preferred_risk_band",
                "preferred_model_predicted_class",
            ],
            prediction_path,
            "ml_prediction_required_columns",
        )

        check_row_count(
            predictions,
            1226,
            prediction_path,
            "ml_prediction_expected_1226_rural_rows",
            tolerance=5,
        )

        check_unique_key(predictions, "dz_code_2022", prediction_path, "ml_prediction_unique_dz_code")

        check_metric_range_0_1(
            predictions,
            ["preferred_model_probability"],
            prediction_path,
            "ml_prediction_probability_range",
        )

        risk_counts = (
            predictions["preferred_risk_band"].value_counts(dropna=False).to_dict()
            if "preferred_risk_band" in predictions.columns
            else {}
        )

        add_result(
            "ml_prediction_risk_band_counts",
            "INFO",
            prediction_path,
            "Risk band distribution in rural prediction outputs.",
            rows=len(predictions),
            columns=len(predictions.columns),
            details={"risk_band_counts": risk_counts},
        )

    cv_path = PROCESSED_DIR / "ml" / "refined_model_results" / "refined_model_cv_summary.csv"
    cv = read_csv_safe(cv_path, "ml_refined_model_cv_summary")

    if cv is not None:
        metric_cols = [
            "accuracy_mean",
            "precision_mean",
            "recall_mean",
            "f1_mean",
            "roc_auc_mean",
        ]

        present_metric_cols = [col for col in metric_cols if col in cv.columns]

        check_metric_range_0_1(
            cv,
            present_metric_cols,
            cv_path,
            "ml_cv_metrics_range",
        )

        add_result(
            "ml_cv_metrics_summary",
            "INFO",
            cv_path,
            "Cross-validation model metrics summary.",
            rows=len(cv),
            columns=len(cv.columns),
            details=cv.to_dict(orient="records"),
        )

    summary_path = PROCESSED_DIR / "ml" / "refined_model_results" / "refined_model_summary.json"
    read_json_safe(summary_path, "ml_refined_model_summary_json", required=False)

    confusion_path = PROCESSED_DIR / "ml" / "baseline_model_results" / "baseline_model_confusion_matrices.csv"
    confusion = read_csv_safe(confusion_path, "ml_baseline_confusion_matrices", required=False)

    if confusion is not None:
        add_result(
            "ml_confusion_matrix_preview",
            "INFO",
            confusion_path,
            "Confusion matrix output loaded for validation evidence.",
            rows=len(confusion),
            columns=len(confusion.columns),
            details=confusion.head(20).to_dict(orient="records"),
        )


# ---------------------------------------------------------------------
# Scenario QA
# ---------------------------------------------------------------------

def qa_scenario_outputs() -> None:
    """
    Validate scenario intervention and simulation outputs.

    Important:
    - scenario_interventions_all.csv is the candidate/intervention design file.
    - scenario_simulation_baseline_all.csv is a wide scenario output file with service-specific
      before/after distance columns.
    - dashboard4_before_after_accessibility_long.csv is the dashboard-ready long version.
    """
    interventions_path = PROCESSED_DIR / "scenario" / "scenario_interventions_all.csv"
    interventions = read_csv_safe(interventions_path, "scenario_interventions_all")

    if interventions is not None:
        require_columns(
            interventions,
            ["dz_code_2022", "recommended_intervention", "intervention_tier"],
            interventions_path,
            "scenario_interventions_required_columns",
        )

        check_row_count(
            interventions,
            409,
            interventions_path,
            "scenario_interventions_expected_409_rows",
            tolerance=10,
        )

        tier_counts = (
            interventions["intervention_tier"].value_counts(dropna=False).to_dict()
            if "intervention_tier" in interventions.columns
            else {}
        )

        intervention_counts = (
            interventions["recommended_intervention"].value_counts(dropna=False).to_dict()
            if "recommended_intervention" in interventions.columns
            else {}
        )

        add_result(
            "scenario_intervention_distribution",
            "INFO",
            interventions_path,
            "Distribution of intervention tiers and recommended intervention types.",
            rows=len(interventions),
            columns=len(interventions.columns),
            details={
                "tier_counts": tier_counts,
                "intervention_counts": intervention_counts,
            },
        )

    simulation_path = PROCESSED_DIR / "scenario" / "scenario_simulation_baseline_all.csv"
    simulation = read_csv_safe(simulation_path, "scenario_simulation_baseline_all")

    if simulation is not None:
        required_simulation_columns = [
            "dz_code_2022",
            "recommended_intervention",
            "intervention_tier",
            "dist_to_nearest_bank_km",
            "dist_to_nearest_atm_km",
            "dist_to_nearest_post_office_km",
            "dist_to_nearest_any_access_point_km",
            "dist_to_nearest_bank_km_after",
            "dist_to_nearest_atm_km_after",
            "dist_to_nearest_post_office_km_after",
            "dist_to_nearest_any_access_point_km_after",
            "bank_km_improvement",
            "atm_km_improvement",
            "post_office_km_improvement",
            "any_access_point_km_improvement",
            "bank_pct_improvement",
            "atm_pct_improvement",
            "post_office_pct_improvement",
            "any_access_point_pct_improvement",
            "total_km_improvement",
            "simulated_site_type",
            "simulated_site_count",
        ]

        require_columns(
            simulation,
            required_simulation_columns,
            simulation_path,
            "scenario_simulation_required_columns",
        )

        check_row_count(
            simulation,
            409,
            simulation_path,
            "scenario_simulation_expected_409_rows",
            tolerance=10,
        )

        check_non_negative_columns(
            simulation,
            [
                "dist_to_nearest_bank_km",
                "dist_to_nearest_atm_km",
                "dist_to_nearest_post_office_km",
                "dist_to_nearest_any_access_point_km",
                "dist_to_nearest_bank_km_after",
                "dist_to_nearest_atm_km_after",
                "dist_to_nearest_post_office_km_after",
                "dist_to_nearest_any_access_point_km_after",
                "bank_km_improvement",
                "atm_km_improvement",
                "post_office_km_improvement",
                "any_access_point_km_improvement",
                "total_km_improvement",
            ],
            simulation_path,
            "scenario_simulation_non_negative_distances_and_improvements",
        )

        check_percentage_range_0_100(
            simulation,
            [
                "bank_pct_improvement",
                "atm_pct_improvement",
                "post_office_pct_improvement",
                "any_access_point_pct_improvement",
            ],
            simulation_path,
            "scenario_simulation_percentage_improvement_range",
        )

        add_result(
            "scenario_simulation_numeric_summary",
            "INFO",
            simulation_path,
            "Scenario simulation numeric summary using service-specific before/after and improvement columns.",
            rows=len(simulation),
            columns=len(simulation.columns),
            details=summarise_dataframe_numeric(
                simulation,
                [
                    "dist_to_nearest_bank_km",
                    "dist_to_nearest_bank_km_after",
                    "bank_km_improvement",
                    "dist_to_nearest_atm_km",
                    "dist_to_nearest_atm_km_after",
                    "atm_km_improvement",
                    "dist_to_nearest_post_office_km",
                    "dist_to_nearest_post_office_km_after",
                    "post_office_km_improvement",
                    "dist_to_nearest_any_access_point_km",
                    "dist_to_nearest_any_access_point_km_after",
                    "any_access_point_km_improvement",
                    "total_km_improvement",
                ],
            ),
        )


# ---------------------------------------------------------------------
# Dashboard export QA
# ---------------------------------------------------------------------

def qa_dashboard_exports() -> None:
    """Validate all dashboard export files used in Power BI."""
    dashboard_exports_dir = PROCESSED_DIR / "dashboard_exports"

    expected_files = [
        # Dashboard 1 exports
        dashboard_exports_dir / "dashboard1_service_counts.csv",
        dashboard_exports_dir / "dashboard1_service_counts_by_ur6.csv",
        dashboard_exports_dir / "dashboard1_rural_zone_summary.csv",

        # Dashboard 2 enhanced exports
        dashboard_exports_dir / "dashboard2_enhanced" / "dashboard2_accessibility_kpis.csv",
        dashboard_exports_dir / "dashboard2_enhanced" / "dashboard2_accessibility_map_ready.csv",
        dashboard_exports_dir / "dashboard2_enhanced" / "dashboard2_distance_distribution.csv",
        dashboard_exports_dir / "dashboard2_enhanced" / "dashboard2_accessibility_ur6_enhanced.csv",
        dashboard_exports_dir / "dashboard2_enhanced" / "dashboard2_vulnerable_population_segments.csv",
        dashboard_exports_dir / "dashboard2_enhanced" / "dashboard2_top_100_vulnerable_access_zones.csv",
        dashboard_exports_dir / "dashboard2_enhanced" / "dashboard2_top_100_least_accessible_rural_zones_enhanced.csv",
        dashboard_exports_dir / "dashboard2_enhanced" / "dashboard2_temporal_bank_change_by_ur6.csv",
        dashboard_exports_dir / "dashboard2_enhanced" / "dashboard2_top_100_worsening_rural_zones.csv",

        # Dashboard 3 enhanced exports
        dashboard_exports_dir / "dashboard3_enhanced" / "dashboard3_model_kpis.csv",
        dashboard_exports_dir / "dashboard3_enhanced" / "dashboard3_model_comparison.csv",
        dashboard_exports_dir / "dashboard3_enhanced" / "dashboard3_preferred_feature_importance_top10.csv",
        dashboard_exports_dir / "dashboard3_enhanced" / "dashboard3_all_model_feature_importance_top10.csv",
        dashboard_exports_dir / "dashboard3_enhanced" / "dashboard3_confusion_matrix_long.csv",
        dashboard_exports_dir / "dashboard3_enhanced" / "dashboard3_confusion_matrix_wide.csv",
        dashboard_exports_dir / "dashboard3_enhanced" / "dashboard3_risk_band_summary.csv",
        dashboard_exports_dir / "dashboard3_enhanced" / "dashboard3_ur6_prediction_summary.csv",
        dashboard_exports_dir / "dashboard3_enhanced" / "dashboard3_underserved_map_ready.csv",
        dashboard_exports_dir / "dashboard3_enhanced" / "dashboard3_top_100_high_risk_zones.csv",

        # Dashboard 4 enhanced exports
        dashboard_exports_dir / "dashboard4_enhanced" / "dashboard4_intervention_kpis.csv",
        dashboard_exports_dir / "dashboard4_enhanced" / "dashboard4_intervention_counts.csv",
        dashboard_exports_dir / "dashboard4_enhanced" / "dashboard4_intervention_tier_summary.csv",
        dashboard_exports_dir / "dashboard4_enhanced" / "dashboard4_ur6_intervention_summary.csv",
        dashboard_exports_dir / "dashboard4_enhanced" / "dashboard4_primary_gap_summary.csv",
        dashboard_exports_dir / "dashboard4_enhanced" / "dashboard4_improvement_distribution.csv",
        dashboard_exports_dir / "dashboard4_enhanced" / "dashboard4_policy_priority_table.csv",
        dashboard_exports_dir / "dashboard4_enhanced" / "dashboard4_top_100_policy_interventions.csv",
        dashboard_exports_dir / "dashboard4_enhanced" / "dashboard4_top_50_policy_interventions.csv",
        dashboard_exports_dir / "dashboard4_enhanced" / "dashboard4_top_20_policy_interventions.csv",
        dashboard_exports_dir / "dashboard4_enhanced" / "dashboard4_intervention_map_ready.csv",
        dashboard_exports_dir / "dashboard4_enhanced" / "dashboard4_before_after_accessibility_long.csv",
    ]

    for file_path in expected_files:
        df = read_csv_safe(file_path, f"dashboard_export_exists_{file_path.stem}", required=True)

        if df is not None:
            add_result(
                f"dashboard_export_shape_{file_path.stem}",
                "INFO",
                file_path,
                "Dashboard export shape captured.",
                rows=len(df),
                columns=len(df.columns),
                details={"columns": list(df.columns)},
            )

    # Coordinate checks for map-ready dashboard outputs.
    map_ready_files = [
        dashboard_exports_dir / "dashboard2_enhanced" / "dashboard2_accessibility_map_ready.csv",
        dashboard_exports_dir / "dashboard3_enhanced" / "dashboard3_underserved_map_ready.csv",
        dashboard_exports_dir / "dashboard4_enhanced" / "dashboard4_intervention_map_ready.csv",
    ]

    for file_path in map_ready_files:
        df = read_csv_safe(file_path, f"dashboard_map_ready_load_{file_path.stem}", required=True)

        if df is not None:
            check_coordinate_range(df, file_path, f"dashboard_map_ready_coordinate_range_{file_path.stem}")


# ---------------------------------------------------------------------
# Output writer
# ---------------------------------------------------------------------

def write_outputs() -> None:
    """Write final QA reports as JSON, CSV, and Markdown."""
    QA_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    result_dicts = [asdict(result) for result in results]

    json_path = QA_DIR / "final_project_qa_report.json"
    json_path.write_text(json.dumps(result_dicts, indent=2), encoding="utf-8")

    csv_path = QA_DIR / "final_project_qa_report.csv"
    pd.DataFrame(result_dicts).to_csv(csv_path, index=False)

    status_counts = pd.Series([r.status for r in results]).value_counts().to_dict()

    fail_count = int(status_counts.get("FAIL", 0))
    warn_count = int(status_counts.get("WARN", 0))
    pass_count = int(status_counts.get("PASS", 0))
    info_count = int(status_counts.get("INFO", 0))

    markdown_lines = [
        "# Final Project QA Report",
        "",
        f"Generated: {timestamp}",
        f"Project root: `{PROJECT_ROOT}`",
        "",
        "## Summary",
        "",
        f"- PASS checks: {pass_count}",
        f"- WARN checks: {warn_count}",
        f"- FAIL checks: {fail_count}",
        f"- INFO records: {info_count}",
        "",
        "## Overall QA judgement",
        "",
    ]

    if fail_count == 0:
        markdown_lines.append("**PASS:** No failing QA checks were detected.")
    else:
        markdown_lines.append(
            "**ACTION REQUIRED:** One or more failing QA checks were detected. "
            "Review the FAIL section before final documentation/export."
        )

    markdown_lines.extend(["", "## Failed checks", ""])

    failed = [r for r in results if r.status == "FAIL"]

    if not failed:
        markdown_lines.append("No failed checks.")
    else:
        for item in failed:
            markdown_lines.append(f"- **{item.check_name}** — {item.message} (`{item.file_path}`)")

    markdown_lines.extend(["", "## Warning checks", ""])

    warned = [r for r in results if r.status == "WARN"]

    if not warned:
        markdown_lines.append("No warning checks.")
    else:
        for item in warned:
            markdown_lines.append(f"- **{item.check_name}** — {item.message} (`{item.file_path}`)")

    markdown_lines.extend(
        [
            "",
            "## Key evidence values",
            "",
            "The full machine-readable QA evidence is stored in:",
            "",
            f"- `{json_path}`",
            f"- `{csv_path}`",
            "",
            "Use these outputs to write the README, pipeline runbook, data validation report, "
            "model evaluation report, scenario simulation method, and limitations section.",
        ]
    )

    md_path = QA_DIR / "final_project_qa_summary.md"
    md_path.write_text("\n".join(markdown_lines), encoding="utf-8")

    print("\n--- Final Project QA Complete ---")
    print(f"PASS: {pass_count}")
    print(f"WARN: {warn_count}")
    print(f"FAIL: {fail_count}")
    print(f"INFO: {info_count}")
    print(f"JSON report: {json_path}")
    print(f"CSV report: {csv_path}")
    print(f"Markdown summary: {md_path}")

    if fail_count > 0:
        print("\nACTION REQUIRED: Review FAIL checks before final submission.")
    else:
        print("\nQA STATUS: PASS - no failing checks detected.")


# ---------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------

def main() -> None:
    """Run all QA checks."""
    print("Starting final project QA...")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Processed data directory: {PROCESSED_DIR}")

    qa_core_geography()
    qa_context_and_accessibility()
    qa_ml_outputs()
    qa_scenario_outputs()
    qa_dashboard_exports()
    write_outputs()


if __name__ == "__main__":
    main()