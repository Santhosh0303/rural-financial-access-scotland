# Repository audit: Rural Financial Access in Scotland

Reviewed 2026-10-02. Repository: [Santhosh0303/rural-financial-access-scotland](https://github.com/Santhosh0303/rural-financial-access-scotland).
Source revision: [b8cbf54fc3027a64d67533dcad799230ab7afee9](https://github.com/Santhosh0303/rural-financial-access-scotland/commit/b8cbf54fc3027a64d67533dcad799230ab7afee9).

**Recommendation:** develop Rural Access Lab as an interactive, budget-constrained service-placement experiment with a service-resilience view. It addresses the original allocation research question and tests which communities are exposed if a single facility disappears. Both capabilities can share one distance engine and use the existing processed data.

This is a completed static review of the 50 files tracked at the revision above: 41 non-empty text files (12,906 lines including whitespace), two DOCX document bodies and seven empty Python package initializers. All tracked folders were inventoried. The DOCX bodies were retrieved as base64, their ZIP content decoded, and their document XML checked against ZIP CRC values. Their rendered layout was not inspected.

**Evidence boundary:** the user's complete raw, intermediate and final files exist on their computer. They are not available to this session. The execution environment failed provisioning, so Python, the pipeline, model fitting and dataset QA were not run. The final dissertation text and PBIX files are not in this tracked tree. The numbers below are documented results, not newly verified observations. This report does not certify all local files or present a refreshed account of Scotland as of October 2026.

## Existing system and strengths

The pipeline has a coherent sequence:

2011/2022 geography -> spatial crosswalk -> population/SIMD context -> OSM services -> representative-point proximity -> rule labels -> baseline and temporal models -> scenario candidates -> illustrative interventions -> dashboard exports -> structural QA.

The strongest reusable parts are the 2022 reporting geography, explicit key/cardinality validation, separate bank/ATM/post-office distances, source/target crosswalk shares, checkpointed OSM extraction, explicit baseline feature allowlist, scaled logistic comparison, out-of-fold logistic threshold sweeps, deterministic temporal nearest-bank tie handling, interpretable intervention rules, and the final README's clear straight-line-distance limitation.

The code is a local dissertation pipeline with substantial documentation. It is not yet a standalone distributable release: the tree has no config package, dependency manifest/lockfile, test directory, workflow, app entry point or license file. There is one main branch and no GitHub release in the recheck before this audit branch.

## Research scope versus implementation

The original dataset/research-question document asks whether evidence-based optimized placement beats non-optimized allocation. It also proposes network travel and change over 2013-2023. See [docs/DATASETS and Research Questions.docx](https://github.com/Santhosh0303/rural-financial-access-scotland/blob/b8cbf54fc3027a64d67533dcad799230ab7afee9/docs/DATASETS%20and%20Research%20Questions.docx). The literature overview similarly positions network-based predictive optimization.

The implemented accessibility functions calculate Euclidean nearest distance. The bank comparison is between year-end 2019 and 2023 snapshots, while the population panel covers 2011-2023 according to the download log. Copying a single current service snapshot across a population panel does not reconstruct yearly service accessibility.

The implemented intervention method zeroes selected distances at each candidate's own origin. It does not solve allocation under a budget or compare allocations. Therefore, a placement experiment would address an original research question that is only partly covered by the current code. The planning Word documents and the final method should be versioned and reconciled; the final thesis text is unavailable, so this report does not judge what that thesis ultimately claims.

"Version 2" in decisions.md already means the temporal extension. Use a public release name such as **Rural Access Lab v2** and keep analytical labels explicit: dissertation baseline, historical bank comparison, new placement experiment and closure stress test.

## Documented results and their interpretation

| Documented item | Value | What can be said now |
| --- | ---: | --- |
| Reporting geography | 7,392 zones | Reported 2022 geography, not a current service observation date |
| Context panel | 96,096 rows | Equals 7,392 x 13 years; key completeness/conservation not reproduced |
| Rural model rows | 1,226 | Reported rural cohort; validate from actual geography |
| Rule-positive rural zones | 289 | Operational p75-derived label, not independently observed exclusion |
| Scenario candidates | 409 | Observed shortlist from a union rule, not a fixed required output |
| Mean nearest bank distance | 4.12 km | Documented Scotland-wide unweighted zone-origin mean |
| Mean nearest ATM/post-office/any distance | 2.34 / 1.48 / 1.15 km | Documented proximity means; services have different capabilities |
| RF CV accuracy/F1/ROC-AUC | .804 / .543 / .793 | Reported random-fold classification evidence |
| RF CV recall | .495 | Roughly half of rule-positive cases identified in reported fold metrics |
| Final QA | 71 PASS, 0 WARN, 0 FAIL | Historical documented structural check result, not rerun here |

Sources: [notebooks/DATA_VALIDATION_REPORT.md](https://github.com/Santhosh0303/rural-financial-access-scotland/blob/b8cbf54fc3027a64d67533dcad799230ab7afee9/notebooks/DATA_VALIDATION_REPORT.md), [notebooks/MODEL_EVALUATION_REPORT.md](https://github.com/Santhosh0303/rural-financial-access-scotland/blob/b8cbf54fc3027a64d67533dcad799230ab7afee9/notebooks/MODEL_EVALUATION_REPORT.md) and [docs/dataset_download_log.txt.txt](https://github.com/Santhosh0303/rural-financial-access-scotland/blob/b8cbf54fc3027a64d67533dcad799230ab7afee9/docs/dataset_download_log.txt.txt).

The reported majority-class accuracy is 937 / 1,226 = **76.43%**. RF's documented 80.4% is about **3.97 percentage points** higher, so accuracy alone overstates the apparent gain. These are arithmetic comparisons of rounded reported numbers, not a statistical significance result. The reported RF and logistic F1 scores differ by only .007; an independent evaluation is needed before calling one robustly superior for policy use.

The 409 high-band zones arise from splitting ranked scores into thirds. They do not establish that one third of rural Scotland exceeds an independently validated risk threshold.

## Confirmed source defects and design limitations

### 1. Dashboard 3 can report probability as kilometres — High, high confidence

Evidence: [src/transform/build_dashboard3_enhanced_exports.py · L444](https://github.com/Santhosh0303/rural-financial-access-scotland/blob/b8cbf54fc3027a64d67533dcad799230ab7afee9/src/transform/build_dashboard3_enhanced_exports.py#L444) substitutes the probability mean when distance fields are absent, but exports it under mean_bank_distance_km and mean_any_access_distance_km.

Upstream, [src/ML/build_model_ready_baseline.py · L109](https://github.com/Santhosh0303/rural-financial-access-scotland/blob/b8cbf54fc3027a64d67533dcad799230ab7afee9/src/ML/build_model_ready_baseline.py#L109) selects a readable table without distance columns. [src/ML/build_prediction_outputs.py · L127](https://github.com/Santhosh0303/rural-financial-access-scotland/blob/b8cbf54fc3027a64d67533dcad799230ab7afee9/src/ML/build_prediction_outputs.py#L127) copies that readable table into predictions. Dashboard 3 does not join accessibility back before calculating the UR6 summary. Under this source path, the fallback is reached.

Impact: a plausible number in the 0-1 range can appear as a distance in km. File existence and probability-range QA do not detect this.

Smallest fix: join the required accessibility columns by dz_code_2022 with 1:1 validation, require complete finite values, and remove the probability fallback. Compare export distance summaries against independently computed summaries from the accessibility table.

### 2. Baseline features select an implicit context year — High, high confidence

Evidence: [src/ML/build_ml_features_baseline.py · L132](https://github.com/Santhosh0303/rural-financial-access-scotland/blob/b8cbf54fc3027a64d67533dcad799230ab7afee9/src/ML/build_ml_features_baseline.py#L132) drops duplicate zone IDs before selecting any year and excludes year from CONTEXT_KEEP_COLS.

The generated context is grouped by zone and year, and the documented panel starts in 2011. The ordinary generated order therefore retains the earliest available row. Regardless of order, the feature snapshot is not explicitly controlled.

Impact: baseline model context can differ from the latest-year context displayed in Dashboard 2 and the 2023 context used by the temporal model.

Smallest fix: require a configured context year, retain it in feature/output metadata, assert one zone-year row per expected zone and rebuild affected features, scores and exports in a versioned output directory.

### 3. Scenario gains follow the zero-distance assumption — High, high confidence

Evidence: [src/scenario/build_scenario_simulation_baseline.py · L67](https://github.com/Santhosh0303/rural-financial-access-scotland/blob/b8cbf54fc3027a64d67533dcad799230ab7afee9/src/scenario/build_scenario_simulation_baseline.py#L67) places each new site at its own origin and explicitly sets relevant distances to 0.

This is a legitimate illustrative self-zone upper-bound experiment if labelled as such. It is not a budget-constrained site-allocation model. It omits neighbour benefits, overlap between sites, competition for a budget, and the effect of adding one site to the common service inventory.

Smallest useful upgrade: evaluate every chosen site against the full rural-origin set and update each origin's nearest relevant-service distance using the shared selected-site set. Compare equal-size allocations. Do not claim a chosen site is commercially feasible.

The sum called total_km_improvement includes "any access" plus service-specific improvements. These overlapping indicators are not four independent journeys; their sum is not total resident travel saved.

### 4. Group elderly-exposure score is inconsistent with zone scores — Medium, high confidence

Evidence: [src/transform/build_dashboard2_enhanced_exports.py · L262](https://github.com/Santhosh0303/rural-financial-access-scotland/blob/b8cbf54fc3027a64d67533dcad799230ab7afee9/src/transform/build_dashboard2_enhanced_exports.py#L262) multiplies the group's older-population sum by an unweighted mean distance. The zone-level counterpart multiplies each zone's older population by its own distance.

For an illustrative two-zone example with older populations 1 and 100 and distances 100 and 1 km, the group formula gives 5,100.5 population-km, but summing the zone-level exposure gives 200. These are synthetic arithmetic values, not Scotland results.

Smallest fix: sum the zone-level products, or use an older-population-weighted mean multiplied by the population sum. Specify population-km as an exposure proxy, not observed trips or travel costs.

### 5. Temporal comparator labels enter the predictor matrix — High for the claimed exclusion, high confidence

Evidence: [src/ML/build_model_ready_temporal.py · L256](https://github.com/Santhosh0303/rural-financial-access-scotland/blob/b8cbf54fc3027a64d67533dcad799230ab7afee9/src/ML/build_model_ready_temporal.py#L256) describes underserved_baseline and critical_underserved_baseline as readable-only comparator labels, but keeps them in readable_df and passes that into build_encoded_table. The encoder drops only id_cols, and [src/ML/train_temporal_models.py · L151](https://github.com/Santhosh0303/rural-financial-access-scotland/blob/b8cbf54fc3027a64d67533dcad799230ab7afee9/src/ML/train_temporal_models.py#L151) drops only the temporal outcome and two auxiliary temporal outcomes.

Impact: these current-snapshot label proxies participate in the temporal model despite the comment. This is not direct algebraic leakage of the bank-change target, but it breaks the intended feature contract and adds post-period information.

Smallest fix: exclude comparator labels explicitly from X and persist a feature list. Add a test that inspects the actual final predictor matrix rather than the intended list alone.

### 6. Temporal analysis is retrospective, not forecasting — High interpretation risk, high confidence

Evidence: [src/ML/build_ml_features_temporal.py · L55](https://github.com/Santhosh0303/rural-financial-access-scotland/blob/b8cbf54fc3027a64d67533dcad799230ab7afee9/src/ML/build_ml_features_temporal.py#L55) selects 2023 context; [src/ML/build_model_ready_temporal.py · L204](https://github.com/Santhosh0303/rural-financial-access-scotland/blob/b8cbf54fc3027a64d67533dcad799230ab7afee9/src/ML/build_model_ready_temporal.py#L204) includes current cross-service distances and closure information extending through 2023.

Impact: random train/test splits can assess retrospective classification of 2019-2023 deterioration. They do not demonstrate that the model could have predicted that deterioration using information available in 2019.

Smallest fix for one day: describe it as retrospective classification. A future forecasting study requires an explicit cutoff, pre-cutoff features, historical inventories and later independent outcome windows.

### 7. Published zone scores are fitted scores; risk bands are relative — High interpretation risk, high confidence

Evidence: [src/ML/build_prediction_outputs.py · L78](https://github.com/Santhosh0303/rural-financial-access-scotland/blob/b8cbf54fc3027a64d67533dcad799230ab7afee9/src/ML/build_prediction_outputs.py#L78) and [src/ML/build_temporal_prediction_outputs.py · L119](https://github.com/Santhosh0303/rural-financial-access-scotland/blob/b8cbf54fc3027a64d67533dcad799230ab7afee9/src/ML/build_temporal_prediction_outputs.py#L119) fit on all scored rural rows. Both band functions rank with method="first" and use qcut into three bands.

Impact: fitted scores are usable for exploratory ranking but must not be presented as new independent validation. Equal score values can be assigned different bands according to input order. No probability calibration is implemented; class weighting further makes direct posterior-risk interpretation uncertain.

Smallest fix: label fitted scores explicitly, produce identity-linked OOF scores for evaluation, separate relative priority thirds from policy thresholds, and define deterministic tie behavior. Use spatially grouped evaluation to test geographic generalization; random nearby-zone splits may be optimistic.

### 8. Threshold metadata can describe a model that was not used — Medium, high confidence

Evidence: [src/ML/build_temporal_prediction_outputs.py · L161](https://github.com/Santhosh0303/rural-financial-access-scotland/blob/b8cbf54fc3027a64d67533dcad799230ab7afee9/src/ML/build_temporal_prediction_outputs.py#L161) uses 0.5 for RF final and policy classifications, while [src/ML/build_temporal_prediction_outputs.py · L263](https://github.com/Santhosh0303/rural-financial-access-scotland/blob/b8cbf54fc3027a64d67533dcad799230ab7afee9/src/ML/build_temporal_prediction_outputs.py#L263) reports the logistic-derived final_threshold and policy_threshold.

Impact occurs if RF becomes the preferred temporal model; the displayed thresholds then disagree with actual classification.

Smallest fix: persist applied thresholds keyed by selected model, and derive metadata from that actual classification path. Avoid silent fallback model selection when required evaluation evidence is absent.

### 9. A scaled-logistic confusion matrix can be sourced from unscaled logistic — Medium, high confidence

Evidence: [src/transform/build_dashboard3_enhanced_exports.py · L281](https://github.com/Santhosh0303/rural-financial-access-scotland/blob/b8cbf54fc3027a64d67533dcad799230ab7afee9/src/transform/build_dashboard3_enhanced_exports.py#L281) aliases the names even though the baseline training script fits unscaled logistic and refined training uses a scaler.

The currently reported preferred RF avoids that alias, but the logistic path is not interchangeable. A held-out RF matrix and RF CV summary can both be shown if their evaluation sources are explicit.

Smallest fix: match exact model configuration, feature snapshot, threshold and split IDs. Persist actual_class and dz_code_2022 for predictions/matrices.

### 10. QA is a structural report, not a release gate — High, high confidence

Evidence: [src/qa/run_final_project_qa.py · L262](https://github.com/Santhosh0303/rural-financial-access-scotland/blob/b8cbf54fc3027a64d67533dcad799230ab7afee9/src/qa/run_final_project_qa.py#L262) coerces invalid values to NaN and counts only negatives, so missing/unparseable distances and positive infinity can pass. [src/qa/run_final_project_qa.py · L631](https://github.com/Santhosh0303/rural-financial-access-scotland/blob/b8cbf54fc3027a64d67533dcad799230ab7afee9/src/qa/run_final_project_qa.py#L631) checks only metric columns that happen to exist. [src/qa/run_final_project_qa.py · L1007](https://github.com/Santhosh0303/rural-financial-access-scotland/blob/b8cbf54fc3027a64d67533dcad799230ab7afee9/src/qa/run_final_project_qa.py#L1007) prints a warning when FAILs exist but does not set a nonzero process exit code.

The code records CRS but does not assert EPSG:27700. It does not check source-to-target population conservation, zone-year uniqueness, distance identities, temporal ML outputs, score calibration or physical feasibility. Map bounds are a broad sanity check, not proof of location accuracy.

Smallest fix: required and finite values, primary/composite keys, expected-cohort coverage from the manifest, CRS assertion, population conservation, any_distance=min(service_distances), before-after improvement identities, complete extraction status and a failing process exit on FAIL. Frozen dissertation row counts should remain snapshot-specific expectations, not universal release constants.

## Data-dependent risks to check locally

These are source-level risks. Their affected counts and practical severity cannot be established without the local files.

| Risk | Source evidence | Required local check |
| --- | --- | --- |
| Incomplete OSM extraction | [src/extract/extract_services_current.py · L253](https://github.com/Santhosh0303/rural-financial-access-scotland/blob/b8cbf54fc3027a64d67533dcad799230ab7afee9/src/extract/extract_services_current.py#L253) catches failures, then combines available checkpoints and prints success; invalid tiles count as completed | Every expected tile has a valid accepted status; no failed/skipped geography masquerades as absence of service; snapshot/query/tile hashes match |
| Service freshness and coverage | Download log leaves OSM extraction dates blank; amenities are a proxy inventory | Record extraction timestamp, source coverage, exclusions, dedup counts, bank ATM-tag handling and verified operational status |
| Historical date uncertainty | [src/accessibility/build_bank_accessibility_temporal.py · L85](https://github.com/Santhosh0303/rural-financial-access-scotland/blob/b8cbf54fc3027a64d67533dcad799230ab7afee9/src/accessibility/build_bank_accessibility_temporal.py#L85) treats missing opening/closing years as unconstrained and does not reconcile closed status | Count missing/contradictory dates, separate known-active from uncertain-active inventories and check against source guidance |
| Population allocation | [src/transform/build_zone_bridge.py · L48](https://github.com/Santhosh0303/rural-financial-access-scotland/blob/b8cbf54fc3027a64d67533dcad799230ab7afee9/src/transform/build_zone_bridge.py#L48) and [src/transform/build_context.py · L343](https://github.com/Santhosh0303/rural-financial-access-scotland/blob/b8cbf54fc3027a64d67533dcad799230ab7afee9/src/transform/build_context.py#L343) use polygon area shares | Source coverage and weight sums, native/harmonized totals by year, older<=total, and sensitivity to population-weighted crosswalks |
| Missing context becomes zero | [src/transform/build_context.py · L379](https://github.com/Santhosh0303/rural-financial-access-scotland/blob/b8cbf54fc3027a64d67533dcad799230ab7afee9/src/transform/build_context.py#L379) sums weighted ranks; [src/transform/build_dashboard2_enhanced_exports.py · L452](https://github.com/Santhosh0303/rural-financial-access-scotland/blob/b8cbf54fc3027a64d67533dcad799230ab7afee9/src/transform/build_dashboard2_enhanced_exports.py#L452) fills population fields with zero | Preserve unknown values, check source missingness, joins, and distinguish observed zero from unknown |
| SIMD rank interpretation | [src/transform/build_context.py · L379](https://github.com/Santhosh0303/rural-financial-access-scotland/blob/b8cbf54fc3027a64d67533dcad799230ab7afee9/src/transform/build_context.py#L379) averages source ranks | Label as area-weighted rank proxies, not official 2022 ranks; do not imply equal rank gaps equal deprivation differences |
| Spatial service assignment | [src/transform/build_service_points_current.py · L122](https://github.com/Santhosh0303/rural-financial-access-scotland/blob/b8cbf54fc3027a64d67533dcad799230ab7afee9/src/transform/build_service_points_current.py#L122) uses within | Count boundary/unmatched points and resolve deterministically; confirm points lie in intended countries and zones |
| Multipoint deduplication/CRS | [src/transform/build_service_points_current.py · L52](https://github.com/Santhosh0303/rural-financial-access-scotland/blob/b8cbf54fc3027a64d67533dcad799230ab7afee9/src/transform/build_service_points_current.py#L52) explodes points then deduplicates by OSM identity; lines 196-204 assume source CRS compatibility | Check whether any multipoints exist and whether valid components are discarded; validate source CRS before constructing the combined layer |
| Period distinct-brand counts | [src/transform/build_bank_closure_change_features.py · L25](https://github.com/Santhosh0303/rural-financial-access-scotland/blob/b8cbf54fc3027a64d67533dcad799230ab7afee9/src/transform/build_bank_closure_change_features.py#L25) sums annual distinct counts | Relabel as summed annual distinct counts or recompute whole-period nunique |
| Cross-source/date comparison | Current service baseline uses OSM; historical banks use Geolytix; origin geography is 2022 | Reconcile providers, eligibility, timestamps and universes before interpreting differences as temporal change |
| Rural scope | [src/transform/prepare_geography.py · L186](https://github.com/Santhosh0303/rural-financial-access-scotland/blob/b8cbf54fc3027a64d67533dcad799230ab7afee9/src/transform/prepare_geography.py#L186) uses only Accessible Rural and Remote Rural | Describe the UR6 rural definition; remote small towns are outside this cohort; do not imply all geographically isolated settlements are included |

A pre/post-COVID comparison is descriptive. This code does not identify a causal effect of COVID, quantify individual financial exclusion, measure actual resident journey times, or establish current opening hours, transport access, service capacity or branch commercial feasibility.

## File-by-file coverage receipt

The line count includes whitespace and final newline conventions. DOCX inspection covered document text, not rendered page layout. Empty initializers were explicitly accounted for.

| Tracked file | Lines / type | Review finding |
| --- | ---: | --- |
| [.gitignore](https://github.com/Santhosh0303/rural-financial-access-scotland/blob/b8cbf54fc3027a64d67533dcad799230ab7afee9/.gitignore) | 51 | Excludes raw/interim/processed data and several dissertation drafts; preserve privacy and add a versioned public input/output manifest. |
| [README.md](https://github.com/Santhosh0303/rural-financial-access-scotland/blob/b8cbf54fc3027a64d67533dcad799230ab7afee9/README.md) | 336 | Clear final proximity-method boundary, but uppercase SRC links, QA CSV and PBIX links do not resolve in this tree; no release or runnable dependency setup is present. |
| [data/README.md](https://github.com/Santhosh0303/rural-financial-access-scotland/blob/b8cbf54fc3027a64d67533dcad799230ab7afee9/data/README.md) | 12 | Explains local data exclusion; does not provide exact downloads, hashes, licenses or a complete reconstruction recipe. |
| [docs/DATASETS and Research Questions.docx](https://github.com/Santhosh0303/rural-financial-access-scotland/blob/b8cbf54fc3027a64d67533dcad799230ab7afee9/docs/DATASETS%20and%20Research%20Questions.docx) | DOCX text | Document text decoded from DOCX with CRC verification. Original scope includes network travel and optimized vs non-optimized allocation; implementation is narrower. |
| [docs/Literature Review Overview 1.docx](https://github.com/Santhosh0303/rural-financial-access-scotland/blob/b8cbf54fc3027a64d67533dcad799230ab7afee9/docs/Literature%20Review%20Overview%201.docx) | DOCX text | Document text decoded with CRC verification. Conceptual positioning claims network/predictive optimization; reconcile this planning text with the implemented proximity method. |
| [docs/dataset_download_log.txt.txt](https://github.com/Santhosh0303/rural-financial-access-scotland/blob/b8cbf54fc3027a64d67533dcad799230ab7afee9/docs/dataset_download_log.txt.txt) | 206 | Official source links and 2026 downloads documented; OSM extraction dates and Which? cross-check remain unfilled/planned, and population ends in 2023. |
| [docs/decisions.md](https://github.com/Santhosh0303/rural-financial-access-scotland/blob/b8cbf54fc3027a64d67533dcad799230ab7afee9/docs/decisions.md) | 64 | Good 2022-reporting/2011-input harmonization decisions and staged QA rule; Version 2 already denotes the temporal extension. |
| [notebooks/DATA_VALIDATION_REPORT.md](https://github.com/Santhosh0303/rural-financial-access-scotland/blob/b8cbf54fc3027a64d67533dcad799230ab7afee9/notebooks/DATA_VALIDATION_REPORT.md) | 375 | Contains reported output counts and summaries, not the underlying data or an independently reproduced QA run. |
| [notebooks/MODEL_EVALUATION_REPORT.md](https://github.com/Santhosh0303/rural-financial-access-scotland/blob/b8cbf54fc3027a64d67533dcad799230ab7afee9/notebooks/MODEL_EVALUATION_REPORT.md) | 225 | Reports metrics, rule-label limitations and recall trade-off; clarify fitted scores, relative bands, baseline advantage and evaluation-source identity. |
| [notebooks/PIPELINE_RUNBOOK.md](https://github.com/Santhosh0303/rural-financial-access-scotland/blob/b8cbf54fc3027a64d67533dcad799230ab7afee9/notebooks/PIPELINE_RUNBOOK.md) | 314 | Useful stage sequence; incomplete executable rebuild recipe, environment versions and snapshot contracts. |
| [notebooks/README.md](https://github.com/Santhosh0303/rural-financial-access-scotland/blob/b8cbf54fc3027a64d67533dcad799230ab7afee9/notebooks/README.md) | 195 | Duplicates overview, says three questions then lists four, leaves a code fence open, and references three absent documentation files. |
| [src/ML/__init__.py](https://github.com/Santhosh0303/rural-financial-access-scotland/blob/b8cbf54fc3027a64d67533dcad799230ab7afee9/src/ML/__init__.py) | 0 | Empty package initializer; inspected. No executable logic or defect identified. |
| [src/ML/build_ml_features_baseline.py](https://github.com/Santhosh0303/rural-financial-access-scotland/blob/b8cbf54fc3027a64d67533dcad799230ab7afee9/src/ML/build_ml_features_baseline.py) | 190 | Zone deduplication keeps first context row without a year filter; derived inverted ranks duplicate information; explicit snapshot selection is needed. |
| [src/ML/build_ml_features_temporal.py](https://github.com/Santhosh0303/rural-financial-access-scotland/blob/b8cbf54fc3027a64d67533dcad799230ab7afee9/src/ML/build_ml_features_temporal.py) | 301 | Uses 2023 context, post-period closure data and current services; inner joins can drop zones; missing labels become zero; describes retrospective classification. |
| [src/ML/build_model_ready_baseline.py](https://github.com/Santhosh0303/rural-financial-access-scotland/blob/b8cbf54fc3027a64d67533dcad799230ab7afee9/src/ML/build_model_ready_baseline.py) | 186 | Explicit feature allowlist removes direct distance/label leakage; numeric missingness and finite values need stricter contracts. |
| [src/ML/build_model_ready_temporal.py](https://github.com/Santhosh0303/rural-financial-access-scotland/blob/b8cbf54fc3027a64d67533dcad799230ab7afee9/src/ML/build_model_ready_temporal.py) | 364 | Comparator labels described as readable-only actually enter the encoded feature table; post-period information remains; missing numeric values become zero. |
| [src/ML/build_prediction_outputs.py](https://github.com/Santhosh0303/rural-financial-access-scotland/blob/b8cbf54fc3027a64d67533dcad799230ab7afee9/src/ML/build_prediction_outputs.py) | 243 | Refits on full rural data and scores those rows; tertiles split equal scores by row order; model-choice fallback can conceal missing evidence. |
| [src/ML/build_temporal_prediction_outputs.py](https://github.com/Santhosh0303/rural-financial-access-scotland/blob/b8cbf54fc3027a64d67533dcad799230ab7afee9/src/ML/build_temporal_prediction_outputs.py) | 361 | Scores training rows; attaches arrays by row position; logistic thresholds are reported even if RF is selected and actually uses 0.5. |
| [src/ML/train_baseline_models.py](https://github.com/Santhosh0303/rural-financial-access-scotland/blob/b8cbf54fc3027a64d67533dcad799230ab7afee9/src/ML/train_baseline_models.py) | 191 | Stratified held-out comparison; unscaled logistic baseline and matrix lacks an explicit persisted actual-class column or zone identity. |
| [src/ML/train_refined_models.py](https://github.com/Santhosh0303/rural-financial-access-scotland/blob/b8cbf54fc3027a64d67533dcad799230ab7afee9/src/ML/train_refined_models.py) | 293 | Scaled five-fold CV and OOF logistic tuning are strengths; spatial holdout, independent threshold evaluation, RF OOF scores and calibration are absent. |
| [src/ML/train_temporal_models.py](https://github.com/Santhosh0303/rural-financial-access-scotland/blob/b8cbf54fc3027a64d67533dcad799230ab7afee9/src/ML/train_temporal_models.py) | 338 | Random splits assess retrospective classification; same complete dataset also informs model choice; test predictions do not retain zone IDs. |
| [src/ML/validate_baseline_models.py](https://github.com/Santhosh0303/rural-financial-access-scotland/blob/b8cbf54fc3027a64d67533dcad799230ab7afee9/src/ML/validate_baseline_models.py) | 212 | OOF logistic threshold sweep; threshold selection and best score use the same OOF labels, requiring separate final evaluation. |
| [src/ML/validate_temporal_models.py](https://github.com/Santhosh0303/rural-financial-access-scotland/blob/b8cbf54fc3027a64d67533dcad799230ab7afee9/src/ML/validate_temporal_models.py) | 244 | OOF threshold sweep is inspectable; matching observed positive prevalence is not a policy objective or probability calibration. |
| [src/__init__.py](https://github.com/Santhosh0303/rural-financial-access-scotland/blob/b8cbf54fc3027a64d67533dcad799230ab7afee9/src/__init__.py) | 0 | Empty package initializer; inspected. No executable logic or defect identified. |
| [src/accessibility/__init__.py](https://github.com/Santhosh0303/rural-financial-access-scotland/blob/b8cbf54fc3027a64d67533dcad799230ab7afee9/src/accessibility/__init__.py) | 0 | Empty package initializer; inspected. No executable logic or defect identified. |
| [src/accessibility/build_accessibility_baseline.py](https://github.com/Santhosh0303/rural-financial-access-scotland/blob/b8cbf54fc3027a64d67533dcad799230ab7afee9/src/accessibility/build_accessibility_baseline.py) | 177 | Computes nearest Euclidean distances and copies one service snapshot across all context years; duplicate metadata gains _x/_y suffixes on panel merge. |
| [src/accessibility/build_bank_accessibility_temporal.py](https://github.com/Santhosh0303/rural-financial-access-scotland/blob/b8cbf54fc3027a64d67533dcad799230ab7afee9/src/accessibility/build_bank_accessibility_temporal.py) | 531 | Consistent 2019/2023 Geolytix snapshots and deterministic nearest tie handling; unknown dates assumed active and current status is ignored. |
| [src/accessibility/build_underserved_baseline.py](https://github.com/Santhosh0303/rural-financial-access-scotland/blob/b8cbf54fc3027a64d67533dcad799230ab7afee9/src/accessibility/build_underserved_baseline.py) | 235 | Rural p75 rule provides operational labels; relative thresholds are not regulatory standards and missing distances can silently become negative labels. |
| [src/accessibility/prepare_accessibility_inputs.py](https://github.com/Santhosh0303/rural-financial-access-scotland/blob/b8cbf54fc3027a64d67533dcad799230ab7afee9/src/accessibility/prepare_accessibility_inputs.py) | 146 | Representative point stays inside polygon but need not represent residents or a feasible site; retain CRS and origin-method metadata. |
| [src/extract/__init__.py](https://github.com/Santhosh0303/rural-financial-access-scotland/blob/b8cbf54fc3027a64d67533dcad799230ab7afee9/src/extract/__init__.py) | 0 | Empty package initializer; inspected. No executable logic or defect identified. |
| [src/extract/extract_bank_closures.py](https://github.com/Santhosh0303/rural-financial-access-scotland/blob/b8cbf54fc3027a64d67533dcad799230ab7afee9/src/extract/extract_bank_closures.py) | 326 | Cleans an already manually compiled spreadsheet; missing closure month becomes January; no implemented Which? reconciliation. |
| [src/extract/extract_services_current.py](https://github.com/Santhosh0303/rural-financial-access-scotland/blob/b8cbf54fc3027a64d67533dcad799230ab7afee9/src/extract/extract_services_current.py) | 423 | OSM tiling/checkpoint resume is useful; failed/skipped tiles do not prevent success and resume lacks snapshot/geometry hashes. |
| [src/qa/__init__.py](https://github.com/Santhosh0303/rural-financial-access-scotland/blob/b8cbf54fc3027a64d67533dcad799230ab7afee9/src/qa/__init__.py) | 0 | Empty package initializer; inspected. No executable logic or defect identified. |
| [src/qa/run_final_project_qa.py](https://github.com/Santhosh0303/rural-financial-access-scotland/blob/b8cbf54fc3027a64d67533dcad799230ab7afee9/src/qa/run_final_project_qa.py) | 1032 | Structural evidence collector; missing/non-finite distances can pass, no CRS/conservation/temporal-ML checks, and recorded FAILs do not set a failing exit code. |
| [src/scenario/__init__.py](https://github.com/Santhosh0303/rural-financial-access-scotland/blob/b8cbf54fc3027a64d67533dcad799230ab7afee9/src/scenario/__init__.py) | 0 | Empty package initializer; inspected. No executable logic or defect identified. |
| [src/scenario/build_scenario_candidates.py](https://github.com/Santhosh0303/rural-financial-access-scotland/blob/b8cbf54fc3027a64d67533dcad799230ab7afee9/src/scenario/build_scenario_candidates.py) | 142 | Priority rule is transparent; shortlist is a union of model/label/tertile selections, so 409 candidates is an observed run count. |
| [src/scenario/build_scenario_interventions.py](https://github.com/Santhosh0303/rural-financial-access-scotland/blob/b8cbf54fc3027a64d67533dcad799230ab7afee9/src/scenario/build_scenario_interventions.py) | 182 | Farthest raw service distance drives recommendation; service-specific thresholds and service-capability/feasibility checks are absent. |
| [src/scenario/build_scenario_simulation_baseline.py](https://github.com/Santhosh0303/rural-financial-access-scotland/blob/b8cbf54fc3027a64d67533dcad799230ab7afee9/src/scenario/build_scenario_simulation_baseline.py) | 266 | Sets own-zone relevant distances to zero and omits neighbour spillovers, shared sites, budgets and allocation comparisons. |
| [src/transform/__init__.py](https://github.com/Santhosh0303/rural-financial-access-scotland/blob/b8cbf54fc3027a64d67533dcad799230ab7afee9/src/transform/__init__.py) | 0 | Empty package initializer; inspected. No executable logic or defect identified. |
| [src/transform/build_bank_closure_change_features.py](https://github.com/Santhosh0303/rural-financial-access-scotland/blob/b8cbf54fc3027a64d67533dcad799230ab7afee9/src/transform/build_bank_closure_change_features.py) | 304 | Annual rates correctly address unequal period lengths; summing yearly distinct brands does not yield distinct brands over the whole period. |
| [src/transform/build_bank_closure_panel.py](https://github.com/Santhosh0303/rural-financial-access-scotland/blob/b8cbf54fc3027a64d67533dcad799230ab7afee9/src/transform/build_bank_closure_panel.py) | 295 | Creates complete 2015-2023 zone-year panel; spatially unmatched closures are reported but not gated; check keys and native event reconciliation. |
| [src/transform/build_context.py](https://github.com/Santhosh0303/rural-financial-access-scotland/blob/b8cbf54fc3027a64d67533dcad799230ab7afee9/src/transform/build_context.py) | 622 | Area allocation is an approximation; lacks mass-conservation gate, averages ordinal SIMD ranks, and pandas sums can turn missing source values into zero. |
| [src/transform/build_dashboard2_enhanced_exports.py](https://github.com/Santhosh0303/rural-financial-access-scotland/blob/b8cbf54fc3027a64d67533dcad799230ab7afee9/src/transform/build_dashboard2_enhanced_exports.py) | 555 | Latest-year context and WGS84 map coordinates are useful; group elderly exposure uses sum(population) times unweighted mean(distance), and missing context becomes zero. |
| [src/transform/build_dashboard3_enhanced_exports.py](https://github.com/Santhosh0303/rural-financial-access-scotland/blob/b8cbf54fc3027a64d67533dcad799230ab7afee9/src/transform/build_dashboard3_enhanced_exports.py) | 683 | Confirmed probability-to-km fallback; missing distances are never joined back; unscaled/scaled logistic alias can mislabel confusion evidence. |
| [src/transform/build_dashboard4_enhanced_exports.py](https://github.com/Santhosh0303/rural-financial-access-scotland/blob/b8cbf54fc3027a64d67533dcad799230ab7afee9/src/transform/build_dashboard4_enhanced_exports.py) | 748 | Illustrative costs and priority scores are export-only; no budget optimizer; missing benefits become zero and clipped after-distances can hide inconsistent improvements. |
| [src/transform/build_dashboard_exports.py](https://github.com/Santhosh0303/rural-financial-access-scotland/blob/b8cbf54fc3027a64d67533dcad799230ab7afee9/src/transform/build_dashboard_exports.py) | 311 | Builds basic page tables with clear keys; requires model/scenario stages even for service overview and accessibility-only exports. |
| [src/transform/build_service_points_current.py](https://github.com/Santhosh0303/rural-financial-access-scotland/blob/b8cbf54fc3027a64d67533dcad799230ab7afee9/src/transform/build_service_points_current.py) | 261 | Converts points/polygons and deduplicates; Multipoints can collapse on OSM ID, within omits boundary matches, and CRS equality needs validation before construction. |
| [src/transform/build_v1_v2_comparison_exports.py](https://github.com/Santhosh0303/rural-financial-access-scotland/blob/b8cbf54fc3027a64d67533dcad799230ab7afee9/src/transform/build_v1_v2_comparison_exports.py) | 530 | Compares distinct targets and different selection rules; add explicit task identity so side-by-side metrics are not described as model improvement. |
| [src/transform/build_zone_bridge.py](https://github.com/Santhosh0303/rural-financial-access-scotland/blob/b8cbf54fc3027a64d67533dcad799230ab7afee9/src/transform/build_zone_bridge.py) | 176 | Correct source/target overlap shares; add weight-sum, complete-coverage and geometry-validity assertions. |
| [src/transform/prepare_geography.py](https://github.com/Santhosh0303/rural-financial-access-scotland/blob/b8cbf54fc3027a64d67533dcad799230ab7afee9/src/transform/prepare_geography.py) | 260 | Uniqueness and rural join coverage guards are strengths; finds first shapefile and assumes projected CRS rather than enforcing it. |

## What should happen next

The selected core direction is an interactive placement experiment. The recommended additional capability is service resilience. The companion [one-day release proposal](./v2-release-proposal-2026-10-02.md) fixes the scope, mathematics, input contract, validation and time budget.

Before execution, connect the real local files to a working workspace. Keep the dissertation snapshot intact and write new outputs into a separately versioned release directory. Do not silently replace historical data or convert hypothetical distances into a current-world claim.

For a minimal prototype, the required local CSVs are:

- data/processed/accessibility/zone_origins_2022.csv
- data/processed/accessibility/service_destinations_current.csv
- data/processed/accessibility/zone_accessibility_baseline_2022.csv
- data/processed/context/zone_year_context_2022.csv (one explicit complete context year may be exported instead)

For deeper reproduction of the confirmed findings, also make config/paths.py, native population/context tables, the zone crosswalk, model feature/readable/encoded tables, evaluation summaries, prediction outputs, original closure inventory, tile log and the final QA evidence available. PBIX files and final thesis text are needed for those separate artifact reviews.

**Completion state:** source audit and release proposal prepared. No product code changed, no dataset recomputed, no tests run, no fix claimed, no app built or release published. Runtime validation and implementation remain dependent on local data and a working executor.
