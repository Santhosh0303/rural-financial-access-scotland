Data Validation Report

Purpose
This report explains how the main project data outputs were checked.
 I wanted to make sure the processed files were present, complete, and safe to use in the dissertation analysis and dashboards.
python -m src.qa.run_final_project_qa

Final QA result
The final QA run passed.
PASS: 71
WARN: 0
FAIL: 0
INFO: 40
No failed checks were found.
No warning checks were found.
The QA evidence files are stored here:
data/processed/qa/final_project_qa_report.json
data/processed/qa/final_project_qa_report.csv
data/processed/qa/final_project_qa_summary.md

What the validation checked
The final QA script checked the main processed files across the project.
It checked:
•	File existence
•	Row counts
•	Required columns
•	Unique Data Zone codes
•	Negative population values
•	Negative distance values
•	Probability ranges
•	Percentage improvement ranges
•	Map coordinate ranges
•	Dashboard export files
This helps show that the outputs were not only created, but also checked before reporting.

Validation summary
Area checked	
Master geography -	PASS
Context and population data	PASS
Accessibility baseline	PASS
Temporal bank accessibility	PASS
ML prediction outputs	PASS
ML model metrics	PASS
Scenario intervention outputs	PASS
Scenario simulation outputs	PASS
Dashboard exports	PASS
Map-ready coordinates	PASS


Core geography validation
File checked
data/processed/geography/zones_master_2022.gpkg







Checks completed
Check	Result
File loaded successfully	PASS
Expected row count	PASS
Unique Data Zone code	PASS
CRS recorded	PASS

Evidence
Rows: 7,392
Columns: 18
CRS: EPSG:27700
Unique key: dz_code_2022


Interpretation
•	The master geography file passed validation.
•	This matters because it is the base layer for the whole project. The accessibility outputs, ML outputs, scenario outputs, and dashboard maps all depend on this geography.
•	The use of EPSG:27700 is suitable because the project uses Scottish spatial data and distance-based analysis.

Context and population validation
File checked
data/processed/context/zone_year_context_2022.csv


Checks completed
Check	Result
File loaded successfully	PASS
Required columns present	PASS
Population values non-negative	PASS
Evidence
Rows: 96,096
Columns: 37
Required columns present: yes
Negative population values: none found

Required columns checked
dz_code_2022
year
population_total
older_population_65_plus
Interpretation
•	The context file passed validation.
•	The file contains population and SIMD-related variables across years. It supports the project because the analysis does not only use location. It also considers deprivation, rural type, population, and older population.

Accessibility baseline validation
File checked
data/processed/accessibility/zone_accessibility_baseline_2022.csv


Checks completed
Check	Result
File loaded successfully	PASS
Expected row count	PASS
Required distance columns present	PASS
Unique Data Zone code	PASS
Negative distance values	PASS




Evidence
Rows: 7,392
Columns: 17
Unique key: dz_code_2022
Negative distance values: none found

Required distance columns checked
1.	dist_to_nearest_bank_km
2.	dist_to_nearest_atm_km
3.	dist_to_nearest_post_office_km
4.	dist_to_nearest_any_access_point_km

Distance summary
Distance field	Mean	Median	Minimum	Maximum
Nearest bank distance	4.12 km	2.12 km	0.01 km	64.89 km
Nearest ATM distance	2.34 km	0.78 km	0.01 km	51.72 km
Nearest post office distance	1.48 km	0.81 km	0.01 km	30.64 km
Nearest any access point distance	1.15 km	0.56 km	0.01 km	30.64 km
Interpretation
The accessibility baseline passed validation.
The results show that post offices and ATMs reduce the average distance to physical financial access when compared with banks alone.

Note
•	These are nearest-distance accessibility results.
•	They should be described as spatial proximity or nearest-distance access.
•	They should not be described as full road-network travel time.



Temporal bank accessibility validation
File checked
data/processed/accessibility/bank_accessibility_temporal_summary_by_ur6.csv
Checks completed
Check	Result
File loaded successfully	PASS
Required columns present	PASS
Negative bank distance values	PASS

Evidence
Rows: 6
Columns: 8
Negative bank distance values: none found
Required columns checked
ur6_name
mean_bank_km_2019
mean_bank_km_2023
mean_bank_change_km
Interpretation
The temporal bank accessibility file passed validation.
This file supports the pre and post COVID part of the project. It helps show how bank access changed by urban rural classification.

Machine learning prediction validation
File checked
data/processed/ml/prediction_outputs/rural_zone_prediction_outputs.csv





Checks completed
Check	Result
File loaded successfully	PASS
Expected rural row count	PASS
Required prediction columns present	PASS
Unique Data Zone code	PASS
Prediction probability range	PASS
Evidence
Rows: 1,226
Columns: 37
Unique key: dz_code_2022
Preferred model probability range: valid from 0 to 1
Required columns checked
dz_code_2022
preferred_model_probability
preferred_risk_band
preferred_model_predicted_class
Risk band counts
Risk band	Count
High	409
Medium	408
Low	409
Interpretation
•	The ML prediction file passed validation.
•	The risk band split is balanced. This is useful for dashboard design because it allows the project to compare high, medium, and low risk rural zones clearly.
•	The model output should still be treated as decision support. It is not a final proof that a zone is underserved.



Machine learning metric validation
Files checked
data/processed/ml/refined_model_results/refined_model_cv_summary.csv
data/processed/ml/refined_model_results/refined_model_summary.json
data/processed/ml/baseline_model_results/baseline_model_confusion_matrices.csv
Checks completed
Check	Result
CV summary loaded	PASS
Model metric ranges valid	PASS
Model summary JSON loaded	PASS
Confusion matrix file loaded	PASS
Model comparison evidence
Model	Accuracy	Precision	Recall	F1	ROC-AUC
Scaled logistic regression	0.721	0.441	0.682	0.536	0.781
Random forest	0.804	0.609	0.495	0.543	0.793
Interpretation
The model metric files passed validation.
Random forest was selected as the preferred model because it achieved stronger accuracy, precision, F1 score, and ROC-AUC.
Logistic regression had higher recall, this shows that the model choice was not perfect in every metric. 









Scenario intervention validation
File checked
data/processed/scenario/scenario_interventions_all.csv
Checks completed
Check	Result
File loaded successfully	PASS
Required intervention columns present	PASS
Expected row count	PASS

Evidence
Rows: 409
Columns: 52
Required columns present: yes
Required columns checked
dz_code_2022
recommended_intervention
intervention_tier
Intervention tier counts
Intervention tier	Count
Tier 1 critical	190
Tier 2 high priority	106
Tier 3 watchlist	113
Recommended intervention counts
Recommended intervention	Count
New bank access candidate	209
Multi-service access candidate	109
New ATM candidate	67
New post office access candidate	24

Interpretation
The intervention candidate file passed validation.
The project identifies 409 rural intervention candidates. This matches the high-risk rural group used in the scenario stage.
The results show that most recommended interventions relate to bank access and multi-service access. This is useful evidence for the scenario dashboard.

Scenario simulation validation
File checked
data/processed/scenario/scenario_simulation_baseline_all.csv
Checks completed
Check	Result
File loaded successfully	PASS
Required simulation columns present	PASS
Expected row count	PASS
Negative distances and improvements	PASS
Percentage improvement range	PASS
Evidence
Rows: 409
Columns: 67
Negative distances: none found
Percentage improvements: valid from 0 to 100
Required columns checked
The QA script checked before and after distance columns for:
bank
ATM
post office
any access point
It also checked improvement columns and simulated site fields.


Scenario numeric evidence
Metric	Mean	Median	Minimum	Maximum
Current bank distance	15.76 km	13.12 km	0.64 km	64.89 km
Projected bank distance	2.17 km	0.00 km	0.00 km	32.19 km
Bank improvement	13.59 km	12.16 km	0.00 km	64.89 km
Current ATM distance	12.99 km	10.78 km	0.48 km	51.72 km
Projected ATM distance	5.82 km	5.34 km	0.00 km	51.72 km
ATM improvement	7.17 km	0.00 km	0.00 km	50.15 km
Current post office distance	7.59 km	6.50 km	0.35 km	30.64 km
Projected post office distance	4.20 km	3.77 km	0.00 km	20.49 km
Post office improvement	3.39 km	0.00 km	0.00 km	30.64 km
Current any access point distance	7.28 km	6.38 km	0.35 km	30.64 km
Projected any access point distance	0.00 km	0.00 km	0.00 km	0.00 km
Any access point improvement	7.28 km	6.38 km	0.35 km	30.64 km
Total improvement	31.42 km	22.61 km	1.10 km	132.18 km
Interpretation
The scenario simulation file passed validation.
The projected any access point distance is 0.00 km because the scenario assumes that a new or improved access point is placed within the candidate Data Zone.

Dashboard export validation
Dashboard 1 exports checked
data/processed/dashboard_exports/dashboard1_service_counts.csv
data/processed/dashboard_exports/dashboard1_service_counts_by_ur6.csv
data/processed/dashboard_exports/dashboard1_rural_zone_summary.csv
Dashboard 2 exports checked
data/processed/dashboard_exports/dashboard2_enhanced/
The QA script checked Dashboard 2 KPI, map, distance, accessibility, vulnerable population, least accessible zone, and temporal bank-change outputs.

Dashboard 3 exports checked
data/processed/dashboard_exports/dashboard3_enhanced/
The QA script checked Dashboard 3 model KPI, model comparison, feature importance, confusion matrix, risk band, UR6 prediction, map-ready, and top high-risk zone outputs.
Dashboard 4 exports checked
data/processed/dashboard_exports/dashboard4_enhanced/
The QA script checked Dashboard 4 KPI, intervention count, tier summary, UR6 summary, policy table, map-ready, top policy intervention, and before and after outputs.
Interpretation
All dashboard export files checked by the QA script loaded successfully.
This means the Power BI dashboard files are supported by validated CSV exports.
Map-ready coordinate validation
The QA script checked map-ready outputs for:
Dashboard 2 accessibility map
Dashboard 3 underserved zone map
Dashboard 4 intervention map
Coordinate validation evidence
Dashboard output	Rows	Latitude range	Longitude range	Status
Dashboard 2 map-ready output	7,392	54.70 to 60.61	-7.47 to -0.85	PASS
Dashboard 3 map-ready output	1,226	54.70 to 60.61	-7.47 to -0.85	PASS
Dashboard 4 map-ready output	409	54.70 to 60.61	-7.47 to -0.85	PASS
Interpretation
The map-ready coordinates passed validation.
The coordinate ranges are inside broad Scotland bounds. This supports the accuracy of the Power BI map visuals.






Validation issues found
The final QA run found no failed checks.
Failed checks: 0
Warning checks: 0
During development, one issue was found in the QA logic itself. The scenario percentage improvement columns were first checked as 0 to 1 values. They were actually stored as 0 to 100 percentage values.
The QA script was corrected. The final run then passed.
This is a useful development note because it shows the validation process was checked and improved.



What this validation proves
The validation proves that:
•	The key processed files exist.
•	The main row counts match expectations.
•	The Data Zone keys are unique where required.
•	Population and distance values do not contain negative values.
•	ML probabilities are in the correct range.
•	Scenario percentage improvements are in the correct percentage range.
•	Map coordinates are suitable for Scotland.
•	Dashboard exports are present and load correctly.

What this validation does not prove
It does not prove that the model gives a final policy truth.
It does not check real-world opening hours, service capacity, road travel time, commercial feasibility, or public transport access.

