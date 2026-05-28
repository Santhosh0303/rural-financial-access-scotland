Rural Financial Access in Scotland


Project overview:



This project studies physical financial access in rural Scotland.

It uses geospatial analysis, machine learning, and scenario simulation to find rural Data Zones that may have weak access to banks, ATMs, and post offices.

The final output is a Power BI dashboard framework with four pages:
1. Financial service distribution overview.
2. Accessibility analysis.
3. Underserved rural zone identification.
4. Scenario simulation and policy insights.

The project was built for an MSc Data Analytics dissertation at De Montfort University.


Main aim:
The aim is to build a reproducible geospatial and machine learning framework that identifies underserved rural zones in Scotland and tests possible intervention options for improving access to physical and financial services.


Research focus:
The project focuses on three main questions:
1. How are physical financial access points distributed across Scotland?
2. Which rural areas have weaker access to banks, ATMs, post offices, or any access point?
3. Can machine learning help identify underserved rural Data Zones?
4. Which rural zones should be prioritised for simulated intervention?



Financial access points used:
The project uses three physical service types:
1.	Banks
2.	ATMs
3.	Post offices


These are combined because rural financial inclusion is not only about bank branches. 

In some rural areas, post offices and ATMs may also support basic access to cash and financial services.


Spatial unit:
The main unit of analysis is the 2022 Scottish Data Zone.
The final QA check confirmed:
•	7,392 Data Zones were loaded in the master geography file.
•	`dz_code_2022` was unique.
•	The master geography used EPSG:27700.
•	Map-ready latitude and longitude values were inside broad Scottish bounds.


Important method note:
This project uses nearest-distance accessibility.
It measures the distance from each Data Zone origin to the nearest bank, ATM, post office, and any financial access point.
This is a spatial proximity method. It should not be described as full road-network travel time unless a later version adds network routing.


Main project pipeline:
The project follows this order:
```text
Raw data
→ Geography preparation
→ Context and population data preparation
→ Financial service point processing
→ Accessibility baseline calculation
→ Bank accessibility change analysis
→ Underserved baseline creation
→ Machine learning feature preparation
→ Model training and prediction
→ Scenario candidate selection
→ Scenario intervention simulation
→ Dashboard export creation
→ Final QA evidence check


Main folders
src/
├── extract/
├── transform/
├── accessibility/
├── ML/
├── scenario/
└── qa/

Folder purpose:

src/extract	Extracts or prepares raw source data
src/transform	Cleans, joins, and prepares analysis and dashboard outputs
src/accessibility	Builds accessibility distance outputs
src/ML	Builds machine learning features, models, and predictions
src/scenario	Builds intervention candidates and scenario simulation outputs
src/qa	Runs final evidence checks across the project


Processed data folders:
data/processed/
├── geography/
├── context/
├── services/
├── accessibility/
├── ml/
├── scenario/
├── dashboard_exports/
└── qa/



Final QA result:
The final QA script was run from the project root using:
python -m src.qa.run_final_project_qa
Final result:
PASS: 71
WARN: 0
FAIL: 0
INFO: 40


The QA output files are stored here:
1.	data/processed/qa/final_project_qa_report.json
2.	data/processed/qa/final_project_qa_report.csv
3.	data/processed/qa/final_project_qa_summary.md


The QA script checked:
•	Core geography files
•	Context and population outputs
•	Accessibility baseline outputs
•	ML prediction outputs
•	ML model metrics
•	Scenario intervention outputs
•	Scenario simulation outputs
•	Dashboard export files
•	Map-ready coordinate ranges


Key validated outputs:
The final QA run confirmed these main row counts:
Output area	Validated row count
Master Data Zone geography	7,392
Context panel	96,096
Accessibility baseline	7,392
Rural ML prediction outputs	1,226
Scenario intervention candidates	409
Scenario simulation outputs	409
Dashboard 4 before and after long table	1,636



Machine learning summary:
The machine learning stage compares two models:
•	Scaled logistic regression
•	Random forest
"The preferred model is random forest."


The final model evidence includes:
•	Cross-validation metrics
•	Feature importance
•	Prediction probabilities
•	Risk bands
•	Held-out confusion matrix output
•	Dashboard-ready ML exports

The model is used as a decision-support tool. It does not prove that a zone is underserved with full certainty. It gives a risk-based classification based on the chosen input data and labels.
Scenario simulation summary

The scenario stage focuses on the highest-risk rural zones.
It produces:
•	Intervention candidates
•	Intervention tiers
•	Recommended intervention types
•	Before and after accessibility estimates
•	Priority benefit scores
•	Relative cost units
•	Dashboard-ready scenario outputs



The scenario outputs are used for policy-style comparison. They are not a final business case. They are an analytical estimate of possible access improvement.
Dashboard pages:
Dashboard 1: Financial Service Distribution Overview
Shows the current financial service mix and distribution across rural and urban classifications.
Dashboard 2: Accessibility Analysis
Shows current access distance patterns, coverage thresholds, vulnerable population exposure, and bank access deterioration.
Dashboard 3: Underserved Rural Zone Identification
Shows machine learning performance, risk bands, feature importance, predicted underserved zones, and model validation evidence.
Dashboard 4: Scenario Simulation and Policy Insights
Shows prioritised intervention candidates, simulated before and after accessibility, intervention tiers, and policy benefit contribution.



How to run final QA:
From the project root:
   cd C:\Dissertation-DMU\rural-financial-access-scotland
   .\.venv\Scripts\Activate.ps1
   python -m src.qa.run_final_project_qa
Expected final result:
FAIL: 0
WARN: 04


Honest limitations:
This project has some limits.
The accessibility method uses the nearest spatial distance. It does not fully model public transport, car travel, road speed, opening hours, service capacity, or user behaviour.
The ML model depends on the quality of the chosen labels and input features.
The scenario simulation is an analytical test. It does not include land availability, cost contracts, bank commercial decisions, or full policy feasibility.





Current project status:
COMPLETE
Current evidence status:
Core pipeline: complete
Dashboard exports: complete
Power BI dashboards: complete
Final QA: passed
Documentation: complete


Related documentation:
The following project documents support this README:
PIPELINE_RUNBOOK.md
DATA_VALIDATION_REPORT.md
MODEL_EVALUATION_REPORT.md



These documents explain the pipeline order, validation results, model evidence, scenario method, final outputs, and project limitations in more detail.


