Pipeline Runbook

Purpose
This document explains how the coding pipeline for the Rural Financial Access in Scotland project was built, checked, and used.

Project location
The documentation files are currently stored in:
C:\Dissertation-DMU\rural-financial-access-scotland\notebooks

Environment
The project was developed in VS Code using a Python virtual environment.
Activate the environment from the project root:
cd C:\Dissertation-DMU\rural-financial-access-scotland
.\.venv\Scripts\Activate.ps1

Main pipeline idea
The pipeline turns raw data into final dashboard-ready outputs.
The full flow is:
Raw data
to geography preparation
to context and population preparation
to service point preparation
to accessibility baseline
to bank accessibility change
to underserved baseline
to machine learning outputs
to scenario simulation
to dashboard exports
to final QA report
Later stages depend on earlier outputs.




Main source folders
src/
├── extract/
├── transform/
├── accessibility/
├── ML/
├── scenario/
└── qa/
Main output folders
data/processed/
├── geography/
├── context/
├── services/
├── accessibility/
├── ml/
├── scenario/
├── dashboard_exports/
└── qa/
Stage 1: Geography preparation
Purpose
This stage prepares the Scottish Data Zone geography.
It creates the main 2022 Data Zone geography file used across the project.
Main output
data/processed/geography/zones_master_2022.gpkg
QA evidence
The final QA check confirmed:
Rows: 7,392
CRS: EPSG:27700
Unique key: dz_code_2022
Status: PASS
This file is the spatial base of the project. Every accessibility, ML, and dashboard output depends on the Data Zone geography being correct.
Stage 2: Context and population preparation
Purpose
This stage joins Data Zones with yearly context information.
It includes:
•	Population
•	Older population
•	Urban rural classification
•	SIMD indicators
•	Active SIMD ranks by year
Main output
data/processed/context/zone_year_context_2022.csv
QA evidence
The final QA check confirmed:
Rows: 96,096
Columns: 37
Required columns present: yes
Negative population values: none found
Status: PASS
The project does not only measure distance. It also looks at rural type, deprivation, population, and older residents. This gives the analysis more value than a simple map.
Stage 3: Financial service point preparation
Purpose
This stage prepares the physical financial access points.
The project includes:
•	Banks
•	ATMs
•	Post offices
These are used because physical financial access in rural areas is not only about bank branches.


Expected service role
1.	Banks support full branch access.
2.	ATMs support cash access.
3.	Post offices support local access to some financial services.
4.	Combining all three gives a fairer view of rural financial inclusion.
Stage 4: Accessibility baseline
Purpose
This stage calculates the nearest access distance from each Data Zone origin to:
•	Nearest bank
•	Nearest ATM
•	Nearest post office
•	Nearest financial access point of any type
Main output
data/processed/accessibility/zone_accessibility_baseline_2022.csv
QA evidence
The final QA check confirmed:
Rows: 7,392
Required distance columns present: yes
Unique Data Zone code: yes
Negative distances: none found
Status: PASS
Key numeric evidence
•	The final QA summary recorded these mean distances:
•	Mean distance to nearest bank: 4.12 km
•	Mean distance to nearest ATM: 2.34 km
•	Mean distance to nearest post office: 1.48 km
•	Mean distance to nearest any access point: 1.15 km
Note:
•	This is a nearest-distance accessibility method.
•	It should be described as spatial proximity or nearest-distance access.
•	It should not be described as full road-network travel time unless a later version adds network routing.


Stage 5: Bank accessibility change analysis
Purpose
This stage compares bank accessibility before and after the post-COVID period.
It gives evidence of bank access deterioration by rural and urban classification.
Main output
data/processed/accessibility/bank_accessibility_temporal_summary_by_ur6.csv
QA evidence
The final QA check confirmed:
Rows: 6
Required columns present: yes
Negative bank distance values: none found
Status: PASS
This stage supports the pre and post-COVID part of the dissertation.
It helps show whether some rural types had weaker bank access after branch closure changes.
Stage 6: Underserved baseline and ML feature preparation
Purpose
This stage creates the modelling table for rural zones.
It prepares the target labels and input features used by machine learning.
The features include:
•	Rural classification
•	Population
•	Older population
•	SIMD ranks
•	SIMD severity features
•	Access-related indicators
Main Interpretation
The model does not replace human judgement.
It gives a risk score that helps identify rural zones that may need attention.



Stage 7: Machine learning modelling
Purpose
This stage trains and compares models for underserved rural zone identification.
The models used are:
•	Scaled logistic regression
•	Random forest
Main outputs
data/processed/ml/prediction_outputs/rural_zone_prediction_outputs.csv
data/processed/ml/refined_model_results/refined_model_cv_summary.csv
data/processed/ml/refined_model_results/refined_model_summary.json
data/processed/ml/baseline_model_results/baseline_model_confusion_matrices.csv
QA evidence
The final QA check confirmed:
Rural prediction rows: 1,226
Required prediction columns present: yes
Unique Data Zone code: yes
Prediction probabilities inside 0 to 1: yes
Status: PASS
Preferred model
The preferred model is:
random_forest
Model evidence
1.	Random forest cross-validation results:
2.	Accuracy: 0.804
3.	Precision: 0.609
4.	Recall: 0.495
5.	F1 score: 0.543
6.	ROC-AUC: 0.793
7.	Scaled logistic regression cross-validation results:
8.	Accuracy: 0.721
9.	Precision: 0.441
10.	Recall: 0.682
11.	F1 score: 0.536
12.	ROC-AUC: 0.781

Model interpretation
Random forest was selected because it had stronger accuracy, precision, F1 score, and ROC-AUC.
Logistic regression had higher recall, so it is still useful as a comparison model. This matters because rural access work may prefer not to miss vulnerable zones.
Risk band output
The rural prediction output contains three risk bands:
High risk: 409
Medium risk: 408
Low risk: 409
Note
•	The model is a decision-support tool.
•	It does not prove that a place is underserved in a final policy sense.
•	It shows risk based on the data and label design used in the project.
Stage 8: Scenario candidate selection
Purpose
This stage selects rural zones that may need intervention.
It uses ML risk output, access gaps, and priority logic to create intervention candidates.
Main output
data/processed/scenario/scenario_interventions_all.csv
QA evidence
The final QA check confirmed:
Rows: 409
Required intervention columns present: yes
Status: PASS
Intervention tiers
The final scenario candidates are split into:
Tier 1 critical: 190
Tier 2 high priority: 106
Tier 3 watchlist: 113
Recommended intervention types
•	The intervention recommendations include:
•	New bank access candidate: 209
•	Multi-service access candidate: 109
•	New ATM candidate: 67
•	New post office access candidate: 24
Stage 9: Scenario simulation
Purpose
This stage estimates how access distance may improve if intervention candidates receive new or improved local access.
Main output
data/processed/scenario/scenario_simulation_baseline_all.csv
QA evidence
The final QA check confirmed:
Rows: 409
Required simulation columns present: yes
Negative distances and improvements: none found
Percentage improvement range: valid from 0 to 100
Status: PASS
Key scenario results
The scenario simulation recorded:
1.	Mean current distance to nearest any access point: 7.28 km
2.	Mean projected distance to nearest any access point: 0.00 km
3.	Mean any access point improvement: 7.28 km
4.	Mean current bank distance: 15.76 km
5.	Mean projected bank distance: 2.17 km
6.	Mean bank improvement: 13.59 km
Note
•	The scenario method is an analytical test.
•	It assumes that a new or improved access point is placed in the candidate zone.
•	It does not claim that a real bank, ATM, or post office can be opened there immediately.
•	It does not include land cost, planning permission, commercial decisions, staff availability, or opening hours.






Stage 10: Dashboard exports
Purpose
This stage creates clean CSV files for Power BI.
The dashboard exports are separated by dashboard page.
Dashboard 1 outputs
data/processed/dashboard_exports/dashboard1_service_counts.csv
data/processed/dashboard_exports/dashboard1_service_counts_by_ur6.csv
data/processed/dashboard_exports/dashboard1_rural_zone_summary.csv
Dashboard 2 outputs
data/processed/dashboard_exports/dashboard2_enhanced/
This folder includes accessibility KPIs, map-ready access data, distance distribution, vulnerable population segments, and temporal bank-change outputs.
Dashboard 3 outputs
data/processed/dashboard_exports/dashboard3_enhanced/
This folder includes model KPIs, model comparison, feature importance, confusion matrix, risk band summary, map-ready underserved zones, and top high-risk zones.
Dashboard 4 outputs
data/processed/dashboard_exports/dashboard4_enhanced/
This folder includes intervention KPIs, intervention counts, tier summaries, policy priority tables, map-ready intervention outputs, and before and after accessibility outputs.
Stage 11: Final QA
Purpose
The final QA script checks whether the main outputs exist and whether the key values are valid.
Command
Run this from the project root:
python -m src.qa.run_final_project_qa
Final result
The final QA run passed:
PASS: 71
WARN: 0
FAIL: 0
INFO: 40
QA output files
data/processed/qa/final_project_qa_report.json
data/processed/qa/final_project_qa_report.csv
data/processed/qa/final_project_qa_summary.md
QA checklist
The QA script checks:
•	Master geography file
•	Data Zone row count
•	Unique Data Zone code
•	Context file columns
•	Population value checks
•	Accessibility baseline columns
•	Distance value checks
•	Temporal bank accessibility output
•	ML prediction output
•	ML metric ranges
•	Scenario intervention output
•	Scenario simulation output
•	Dashboard export files
•	Map-ready coordinate ranges
What not to rerun without reason
Do not rerun heavy extraction steps unless needed.
OSM or external data extraction may take time and may return slightly different results if the live source has changed.
Final runbook judgement
The final QA run shows that the main processed files, ML outputs, scenario outputs, and dashboard exports are present and valid.

