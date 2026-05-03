Model Evaluation Report

Purpose
The model was built to support the identification of underserved rural Data Zones in Scotland.
It is not used as final proof by itself. It is used as a decision-support method. The final judgement also uses accessibility results, rural context, and scenario analysis.

Model task
The model task is binary classification.

It predicts whether a rural Data Zone is likely to be underserved.

The target variable is: “underserved_baseline.”
The model output is used to create:
•	Prediction probabilities
•	Predicted classes
•	Risk bands
•	Feature importance
•	Dashboard 3 outputs
•	Scenario candidate inputs
Model evidence files
The main model files are:
data/processed/ml/prediction_outputs/rural_zone_prediction_outputs.csv
data/processed/ml/refined_model_results/refined_model_cv_summary.csv
data/processed/ml/refined_model_results/refined_model_summary.json
data/processed/ml/baseline_model_results/baseline_model_confusion_matrices.csv
The final QA script confirmed that these files loaded successfully.
Final QA result
The final QA run passed.
PASS: 71
WARN: 0
FAIL: 0
INFO: 40

Modelling dataset
The refined model used the rural modelling dataset.
Final model summary:
Rows: 1,226
Features: 26
Target distribution:
Class	Meaning	Count
 0	Not underserved	937
1	Underserved	289

Class balance note
The target is not evenly balanced.
There are more non-underserved zones than underserved zones.
This matters because accuracy alone can be misleading. A model can get high accuracy by mostly predicting the larger class.
For this reason, the model was compared using several metrics:
•	Accuracy
•	Precision
•	Recall
•	F1 score
•	ROC-AUC
Models compared
Two models were compared:
Model	Reason for using it
Scaled logistic regression	Clear baseline model. Easier to explain. Useful for comparison.
Random forest	Handles non-linear patterns better. Can show feature importance.

Cross-validation results
The model comparison used cross-validation.
Model	Accuracy	Precision	Recall	F1 score	ROC-AUC
Scaled logistic regression	0.721	0.441	0.682	0.536	0.781
Random forest	0.804	0.609	0.495	0.543	0.793
Preferred model
The preferred model is:
random_forest
Why random forest was selected
Random forest was selected because it gave stronger results on most key metrics.
It had better:
•	Accuracy
•	Precision
•	F1 score
•	ROC-AUC
The result was:
•	Random forest accuracy: 0.804
•	Random forest precision: 0.609
•	Random forest F1 score: 0.543
•	Random forest ROC-AUC: 0.793
These values show that random forest gave the strongest overall model performance.

Why logistic regression is still useful
Logistic regression had lower accuracy and precision.
But it had higher recall:
•	Logistic regression recall: 0.682
•	Random forest recall: 0.495
In this project, missing an underserved rural zone can be a problem. A high recall model finds more possible underserved zones, but it may also create more false positives.
So logistic regression is useful as a comparison model. It shows the trade-off between finding more underserved zones and being more precise.

Main model trade-off
The model choice is not perfect.
Random forest is more precise. It is better when the aim is to focus on stronger candidates.
Logistic regression has higher recall. It is better when the aim is to avoid missing possible underserved areas.
For this dissertation, random forest was chosen because the scenario stage needs a smaller and clearer high-risk group for intervention planning.
Prediction output validation
The final prediction output is:
data/processed/ml/prediction_outputs/rural_zone_prediction_outputs.csv
Final QA checks confirmed:
Rows: 1,226
Columns: 37
Unique Data Zone code: yes
Preferred model probability range: valid from 0 to 1
Status: PASS
Required columns checked:
dz_code_2022
preferred_model_probability
preferred_risk_band
preferred_model_predicted_class
Risk band output
The prediction output was split into three risk bands.
Risk band	Count
High	409
Medium	408
Low	409
Why risk bands were used
Risk bands make the results easier to explain in the dashboard.
They also help the scenario stage.
The high-risk group was used as the main pool for intervention candidates.
Model threshold evidence
The refined model summary also stored threshold evidence for scaled logistic regression.
Best scaled logistic threshold by F1:
•	Threshold: 0.60
•	Accuracy: 0.773
•	Precision: 0.517
•	Recall: 0.578
•	F1 score: 0.546
Best scaled logistic threshold by recall:
•	Threshold: 0.30
•	Accuracy: 0.552
•	Precision: 0.328
•	Recall: 0.858
•	F1 score: 0.475
This shows a clear trade-off.
Lowering the threshold increases recall. But it reduces precision and accuracy.
This is useful evidence because it shows the model was not treated as a black box.
Confusion matrix evidence
The project also saved confusion matrix evidence.
File checked:
data/processed/ml/baseline_model_results/baseline_model_confusion_matrices.csv
The final QA script confirmed this file loaded successfully.
The saved values were:
Model	Predicted 0	Predicted 1
logistic_regression	131	57
logistic_regression	22	36
random_forest	167	21
random_forest	35	23
Confusion matrix interpretation
The confusion matrix shows that the models behave differently.
Logistic regression predicts more positive underserved cases.
Random forest is more conservative and predicts fewer positive cases.
This fits the metric results.
Logistic regression has higher recall. Random forest has better precision and overall accuracy.
Feature importance
The random forest model was also used for feature importance.
Feature importance helps explain which variables are most useful for prediction, and this is useful for Dashboard 3 
The dashboard export file is:
data/processed/dashboard_exports/dashboard3_enhanced/dashboard3_preferred_feature_importance_top10.csv
The QA script confirmed this file loaded successfully.
Why feature importance matters
Feature importance supports interpretation.
It helps connect the model result to real project themes such as:
•	Rurality
•	Deprivation
•	Access burden
•	Older population
•	Socioeconomic context
Dashboard 3 model outputs
The model stage supports Dashboard 3.
The checked dashboard outputs include:
1.	dashboard3_model_kpis.csv
2.	dashboard3_model_comparison.csv
3.	dashboard3_preferred_feature_importance_top10.csv
4.	dashboard3_all_model_feature_importance_top10.csv
5.	dashboard3_confusion_matrix_long.csv
6.	dashboard3_confusion_matrix_wide.csv
7.	dashboard3_risk_band_summary.csv
8.	dashboard3_ur6_prediction_summary.csv
9.	dashboard3_underserved_map_ready.csv
10.	dashboard3_top_100_high_risk_zones.csv
All these files loaded successfully in the final QA run.




How the model supports the research question
The model supports this research question:
To what extent can geospatial and socioeconomic indicators predict underserved rural zones in Scotland at small-area level?
The answer is cautious but useful. The model can identify risk patterns at Data Zone level with fair performance.
Random forest reached:
ROC-AUC: 0.793
Accuracy: 0.804
This suggests the input features contain useful signals.
But the model is not perfect.
Recall is only:
0.495
So the model may miss some underserved zones.
This means the model should support decision-making, not replace local review or policy judgement.
Strengths of the model
The model has several strengths:
•	It uses small-area rural data.
•	It combines geospatial and socioeconomic features.
•	It compares two models.
•	It uses cross-validation.
•	It reports several metrics, not only accuracy.
•	It creates map-ready prediction outputs.
•	It links directly to the scenario simulation stage.
Weaknesses of the model
The model also has limits:
•	The target label is based on project rules, not direct survey evidence.
•	The data is imbalanced.
•	Recall is not very high for the preferred model.
•	Some rural access issues may not be captured by the available data.
•	The model does not know about opening hours, public transport, digital banking ability, or local service quality.
Why is this approach still useful
The aim is to build a working geospatial decision-support method.
The model helps move the project beyond basic maps.
It gives a structured way to rank rural zones and select candidates for scenario testing.
Final Policy Judgement
The model is good enough for dissertation decision support.
It is not good enough to be used alone for real public policy.
The best use is this:
Use the model to identify likely high-risk rural zones.
Then combine it with accessibility maps, scenario outputs, local evidence, and policy review.

