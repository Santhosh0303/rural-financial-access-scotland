# Rural Access Lab v2: placement and service resilience

Design draft, 2026-10-02. Companion: [repository audit](./repository-audit-2026-10-02.md).
Based on dissertation source revision [b8cbf54](https://github.com/Santhosh0303/rural-financial-access-scotland/commit/b8cbf54fc3027a64d67533dcad799230ab7afee9).

## The proposed contribution

Turn the dissertation into an experiment people can operate: **with a budget of k new sites, where does placement reduce rural access burden most, and what happens if one existing facility disappears?**

The user selected a placement capability plus an additional critical, unconventional and underrated value. The recommended addition is **continuity of access**: identify communities that have a nearby service but little alternative if that service is lost. Average proximity alone can hide this dependence.

This extends the dissertation's original question about optimized versus non-optimized allocation. It also introduces a distinct test of fragility. It is a defensible research extension; a claim of new-to-the-world methodological novelty would require a separate literature review.

Two linked experiments share the same coordinates, weights and distance calculations:

1. **Place services:** choose a site budget, compare a greedy placement heuristic with simple priority and seeded random allocations, and inspect benefits across all rural origins.
2. **Stress access:** remove one existing facility, see the increase in modeled burden, then compare how new sites reduce that burden.

The first release is a transparent, local-first Python app. Codex or Claude Code can assist with implementation and verification. There is no need for a runtime language model to calculate distances or choose sites.

## One-day scope

Start with **bank access only**. Use all accepted existing banks in the frozen Scotland inventory as destinations and the dissertation's UR6 Accessible Rural/Remote Rural cohort as origins. Urban banks can serve rural origins in the model. Remote small towns remain outside this cohort unless the research population is explicitly expanded.

Use one explicitly selected, complete context year, initially 2023 if the local table confirms coverage. Keep geography vintage, population year and service observation date as separate metadata. A filename containing 2022 denotes reporting geography, not the date of every observation.

Candidate locations are the rural representative points already produced by the pipeline. They are **hypothetical siting points**, not verified properties or commercially feasible locations. Evaluate all eligible rural candidates, rather than only the historical 409-zone shortlist. Collapse co-located candidate points to one siting option with a stable representative ID and record the origin-to-candidate mapping. A future eligible-site CSV can replace this candidate pool.

A budget means a **number of sites**, assuming equal site cost. Initial presets are 0, 1, 3 and 5; allow up to 10 only after measuring runtime. No monetary return, construction estimate, capacity or branch viability is inferred.

The user can choose population_total or older_population_65_plus as the objective weight. Keep them as separate objectives. Show tail-distance and closure-exposure measures alongside the total benefit so the concentration of benefit remains visible.

Scope boundaries:

- Straight-line distances between zone representative points and facility points, in kilometres; no road, ferry, travel-time or transport-availability claim.
- One selected facility loss at a time; no national simultaneous-shock model or closure-likelihood forecast.
- Greedy allocation with comparisons; no claim of globally optimal, legally feasible or commercially viable placement.
- Frozen source data and a recorded snapshot; no implication that the inventory describes October 2026.
- A bank is not interchangeable with an ATM or a post office. Other service types can be added later with explicit capability definitions.
- Existing models and dissertation results remain separately identified until reproduced. Model retraining is not a prerequisite for this distance-based experiment.

These boundaries keep an eight-hour prototype plausible once the local data and a working environment are available. They are not a promise of an eight-hour validated production deployment.

## Scientific contract

### Shared-site placement

Let x_i be origin i, E the accepted existing bank inventory, and c_j candidate j. All coordinates are verified EPSG:27700 metres. Convert Euclidean distances to km by dividing by 1,000.

Baseline distance:

    d_i = min over e in E of distance(x_i, e)

Distance after placing the shared set S:

    d_i(S) = min(d_i, min over j in S of distance(x_i, c_j))

For the selected nonnegative population weight w_i, maximize:

    F(S) = sum_i w_i * (d_i - d_i(S)), subject to |S| = k

F is a **population-km reduction in a zone-origin exposure proxy**. It is not observed journeys saved, resident travel distance, money saved or financial inclusion measured at individual level. The weight of every resident in a zone is attached to that zone's one representative point.

At each step, select the unused candidate with the largest additional F given sites already selected. Break ties deterministically by candidate ID. Recompute marginal gains after every addition, so overlapping catchments are counted correctly. Keep exactly k eligible distinct sites, including zero-gain additions if necessary, and show their zero incremental benefit. Reject k greater than the eligible pool. k=0 returns the baseline.

The heuristic is deterministic for identical inputs, objective and tie rule. It has diminishing marginal benefits under this objective, but the release does not call its result globally optimal. For k=1, verify it against exhaustive evaluation of every candidate.

A site at an origin can legitimately make that origin's modeled distance zero. With a shared finite budget, it does not automatically make every other origin's distance zero.

### Allocation comparisons

Use the same k, candidate pool, accepted service inventory, origin cohort, context year and objective for each allocation:

| Allocation | Definition |
| --- | --- |
| Baseline | No sites added; reference burden |
| Greedy | Sequential maximum marginal reduction |
| Distance priority | Candidates sorted by existing nearest-bank distance, then stable candidate ID |
| Random | 100 allocations sampled without replacement with a recorded seed |
| Optional legacy priority | Historical priority ranking only if all candidate scores and its data provenance are available and explicitly labelled |

Distance priority is available without trusting the legacy fitted model scores. Do not manufacture a legacy comparison when scores are absent. Restricting a comparison to a different candidate pool makes it a different experiment and must be labelled separately.

Report the greedy benefit, distance-priority benefit, random distribution and the fraction of random draws the greedy result exceeds. Do not promise that greedy beats every draw or that a positive difference is a causal or statistically generalizable policy effect.

Report both weighted means and the unweighted p90/worst zone distance. p90 is the 90th percentile of rural origin distances using a declared quantile convention, initially NumPy's linear method. It measures the distribution of zone origins, not the population distribution of actual journeys.

A configurable illustrative threshold, initially 10 km, can show the number of origins and sum of zone population associated with origins at or below the threshold. Call it an illustrative proximity threshold, not an official access standard. Do not describe that proxy as the exact number of residents within 10 km.

### Facility loss and fragility

For a selected existing facility e*, define:

    d_i(loss, S) = min distance from x_i to (E excluding e*) union S

Compare the same placement S before and after loss:

    loss_increase_i = d_i(loss, S) - d_i(S)

Expose two different quantities:

- **Loss burden for S:** sum_i w_i * loss_increase_i.
- **Absolute improvement during the same loss:** sum_i w_i * (d_i(loss, empty) - d_i(loss, S)).

Keep both because a small loss increment can coexist with poor absolute access. An origin whose nearest bank is not removed may have zero additional loss burden while still being far from every bank. Compare each quantity like-for-like.

Show affected zone count, affected population and older population under the representative-point assumption, mean/p90 changes and origins crossing the chosen threshold. These are simulated consequences of removing an inventory point, not a forecast of which bank will close or evidence of residents' actual bank choice.

For the unaugmented baseline, cache the two nearest **distinct facility IDs** per origin. Losing the nearest sends the origin to its second-nearest; losing any other facility leaves its distance unchanged. For augmented scenarios, include selected sites in the calculation. Validate the cached result against direct recomputation.

Fragility gap is second-nearest minus nearest distance. A large gap indicates little nearby alternative in this proximity model. A nearest-bank assignment is a model catchment, not an observed customer catchment.

Facilities sharing coordinates remain separate inventory assets when the source supports separate IDs. Losing one co-located asset can leave distance unchanged. This does not simulate the loss of an entire building or shared infrastructure. Resolve duplicate source records before modeling.

With only one accepted facility and no replacement, show an explicit unserved state and count. Never replace no-alternative distance with zero. Finite-only mean summaries cannot silently drop unserved origins; flag the count and mark the overall distance summary unavailable.

Rank facility-loss scenarios by measured modeled burden as a diagnostic. Do not label the ranking a closure probability. A full minimax or stochastic resilience optimizer is a later study; day one evaluates selected losses and can rerun the placement heuristic on the remaining inventory.

## A small, checkable example

This is a synthetic one-dimensional fixture, not a Scotland result. Coordinates are written in km for readability; implementation inputs use metres.

| Origin | x (km) | Total population | Older population | Baseline nearest-bank distance (km) |
| --- | ---: | ---: | ---: | ---: |
| R1 | 0 | 100 | 1 | 10 |
| R2 | 10 | 10 | 1 | 20 |
| R3 | 30 | 100 | 100 | 10 |

Existing banks are at -10 and 40 km. Candidates are the three origins.

With one site and total-population weights, R1 has benefit 1,100 population-km, R2 has 200 and R3 has 1,000. The greedy choice is R1. With older-population weights the benefits are 20, 20 and 1,000, so the choice is R3.

Losing the existing bank at 40 km changes baseline distances from [10, 20, 10] to [10, 20, 40]. Total-population loss burden is 3,000 population-km. With a new site at R1, normal distances are [0, 10, 10] and loss distances [0, 10, 30], giving a loss burden of 2,000. Placing at R3 instead makes its nearest distance zero both before and after loss.

This demonstrates why total-population benefit, older-population benefit and loss exposure answer different questions. The product must show which objective was selected.

## Inputs and release manifest

The four minimal existing CSVs are:

| Path | Required fields/use |
| --- | --- |
| data/processed/accessibility/zone_origins_2022.csv | dz_code_2022, origin_x, origin_y, is_rural; names and UR6 labels for display |
| data/processed/accessibility/service_destinations_current.csv | osm_element_type, osm_id, service_type, dest_x, dest_y; bank subset supplies existing assets |
| data/processed/accessibility/zone_accessibility_baseline_2022.csv | dz_code_2022, dist_to_nearest_bank_km; reconcile against recomputation from the same inventory |
| data/processed/context/zone_year_context_2022.csv | dz_code_2022, year, population_total, older_population_65_plus; one explicit complete year |

Additional CRS evidence is essential: supply the corresponding origin/destination GeoPackages or verifiable export metadata from their source layers. CSV coordinates alone do not prove EPSG:27700.

Create a release manifest with:

- File SHA-256 hashes, relative paths, schema version and accepted row counts.
- Source and extraction dates; preserve unknown dates explicitly.
- Geography vintage, context year, UR6 cohort definition and coordinate system.
- Accepted/excluded destination rules, duplicate handling, extraction coverage status and unique facility IDs.
- Candidate definition, objective, threshold, site budget, random seed and algorithm version.
- Source code commit and actual dependency versions used for the run.

Use facility identity from source + osm_element_type + osm_id + service_type. Validate its uniqueness after approved deduplication. Do not use a volatile dataframe row number as identity.

Reject missing/unparseable/nonfinite coordinates, invalid flags, duplicate origin IDs, duplicate facility IDs, missing selected-year context, negative populations, older population above total, and all-zero selected weights. Zone population allocations may be fractional and are not automatically invalid.

Require one accepted context row per origin for the selected year and a validated 1:1 join. Do not pick the first available year or fill unknown population with zero. Extra non-cohort rows may be reported and filtered deliberately.

Recompute bank distances from the accepted inventory before optimization. Reconcile the existing bank-distance CSV only when provider, filter, coordinates and snapshot match. Use a declared tolerance based on source export precision; do not blindly accept a mismatch as rounding. If provenance differs, display it as a separate baseline and use the internally recomputed bank baseline for all experiment comparisons.

If extraction completeness or snapshot provenance is unknown, retain an explicit exploratory status. Do not certify real-world coverage by passing structural checks. Run the deterministic mathematical prototype only with clearly stated assumptions; do not label it a verified current inventory.

## Architecture and integration

Use the existing Python project with **Streamlit, NumPy/pandas, SciPy and Plotly**, plus pyproj for coordinate conversion if GeoPandas is not needed at runtime. Record versions from the working environment instead of inventing pins before installation. A tested lockfile or equivalent resolved environment belongs in the release.

Proposed new components:

| Component | Responsibility |
| --- | --- |
| src/scenario/service_placement.py | Pure distance, benefit, greedy and comparison functions |
| src/scenario/service_resilience.py | Facility loss, nearest alternatives and exposure calculations |
| src/qa/validate_release_inputs.py | Explicit contracts and nonzero CLI exit on failure |
| app.py | Local Streamlit interface using the shared functions |
| tests/test_service_experiments.py | Independent synthetic fixtures and invariants |
| docs/RURAL_ACCESS_LAB_METHOD.md | Final implemented method and definitions |
| release manifest | Frozen input and run identity |

Read processed files directly with a user-provided data root. Avoid making the app import unavailable config.paths merely to open existing CSVs. Make any portable config repair separately and verify existing pipeline commands before claiming a full rebuild.

Cache inputs by hashes and distance matrices by inventory/candidate/cohort identity. A reported cohort of 1,226 origins and 1,226 candidates produces about 1.5 million pairwise distances; actual cohort size and memory use must be measured. Optimize in NumPy and cache random comparisons rather than rerunning Python nested loops on every UI event.

Convert EPSG:27700 to WGS84 only for map display. Distances continue to use the verified projected coordinates. The same engine drives map colors, charts and exported tables.

Write new artifacts under data/processed/release_v2/ without silently overwriting the frozen dissertation outputs. Produce:

- Per-origin before/after/loss distances and objective weights.
- Selected-site order and marginal/cumulative benefit.
- Comparison draws and their summary.
- Facility-loss summary and origin dependence table.
- Input/run manifests and machine-readable QA results.

Export missing alternatives explicitly. JSON uses null plus status/count fields rather than nonstandard Infinity values.

## Interface people can use

At the top, show the experiment's population year, service snapshot status and straight-line method. These details affect how people should interpret the result.

The placement view offers site budget and population objective controls, a before/after map, selected sites, weighted mean and p90 distances, and a compact allocation-comparison chart. Selecting a site exposes its marginal benefit and neighbouring origins helped.

The resilience view lets the user choose an existing facility, shows its modeled nearest-origin catchment and backup-distance gap, and displays how the same site budget changes access after losing that facility. A threshold-crossing view makes a small number of highly affected origins visible.

Every result should have a clear source snapshot and downloadable table. Use plain definitions next to the numbers. Explain population-km once. Do not present estimated minutes, named household impacts or unsupported savings.

A deterministic explanation can describe the largest calculated change and selected objective. A conversational assistant is optional later and would cite the exported calculation records; it should not invent placement results.

## Blocking findings and repair scope

The [audit](./repository-audit-2026-10-02.md) documents ten confirmed defects or interpretation limitations and further data-dependent checks.

For this release, the blocking priorities are:

1. **Distance units and provenance:** use actual bank distances; remove the Dashboard 3 probability-to-km fallback if reusing that export. The new app should not use that summary as a baseline.
2. **Explicit context year:** select and retain the chosen year before joining weights. If legacy baseline models are regenerated, fix their implicit first-year selection and version affected outputs.
3. **Finite values and keys:** implement release input QA that fails the process, checks CRS evidence and rejects missing/mismatched rows.
4. **Shared-service scenarios:** use the common inventory and budget contract above rather than the self-zone zero-distance scenario as optimized allocation.

Dashboard 2's group exposure formula should be repaired if reused, by summing population-times-distance at the zone level. The app can calculate it directly from its verified origin table.

Temporal predictor exclusions, threshold identity, scaled/unscaled model identity, calibrated/OOF scores and spatial validation matter for a future research rebuild. Keep affected legacy material labelled and outside the new app's policy claims. Do not claim all ten findings have been repaired in a one-day distance prototype.

## Eight-hour implementation sequence

The clock starts when the local data are attached to a working environment. Stop at a blocking data-contract failure and report the failing rows; do not silently synthesize an accepted input.

| Time | Deliverable | Evidence |
| --- | --- | --- |
| 0-1 h | Inspect four CSVs plus CRS metadata; select context year; freeze hashes; restore portable runtime setup | Accepted schema/keys/cohort, input manifest, installed dependency versions |
| 1-2 h | Independent bank baseline and release input QA; correct any reused unit/year/exposure path | Baseline reconciliation and explicit QA failures on invalid fixtures |
| 2-4 h | Shared-site greedy placement, seeded comparisons, facility-loss engine | Hand-computed fixtures, k=1 exhaustive comparison, direct versus cached closure calculations |
| 4-6 h | Streamlit controls, before/after and loss maps, comparison chart, export | Local app opens with real inputs; interactions produce the same table calculations |
| 6-7 h | Complete invariant checks and real-data smoke run | Test output, data QA exit status, measured memory/runtime, no silent missing states |
| 7-8 h | Run k=0/1/3/5 experiments, preserve artifacts, document method and local start command | Reproducible manifests/results, reviewed UI capture, draft release notes |

Spend the day on this coherent slice. Nationwide transport routing, fresh inventory scraping, every legacy-model repair and a production hosting pipeline need separate time.

## Validation before claiming a release

Meaningful checks must establish:

- k=0 reproduces the internally recomputed bank baseline.
- Adding a site cannot increase any origin distance; loss cannot reduce it.
- k=1 greedy matches exhaustive candidate evaluation on independent fixtures.
- Marginal gains account for overlap; cumulative benefit equals the final weighted distance difference.
- Stable IDs and fixed seed produce identical allocations and comparison results after row reordering.
- Every comparison uses equal site count, eligible pool, weights and snapshot.
- Two-nearest closure caching matches direct removal, including equal distances and co-located distinct assets.
- Unknown/no-alternative access stays explicit and does not enter a zero-distance or finite-only silent fallback.
- Unit conversion is checked against known metre coordinates; invalid numbers and duplicate keys fail QA.
- Explicit-year joins remain invariant to input sorting and reject missing coverage.
- Older population is bounded by total population; weighted means equal sum-products divided by weight sum.
- Actual local bank-distance recomputation reconciles with the accepted inventory, not a probability field.
- UI summaries and CSV exports match the same engine output.
- Required test and QA commands pass with recorded output. No pass claim is made from source inspection alone.

The tests above are proposed acceptance checks, not tests already written or executed. Source review and a small JavaScript arithmetic check are not substitutes for running the Python implementation on the real files.

## Research outputs

A concise release note can present these questions without pre-claiming their answers:

1. How much modeled rural bank-access burden can k sites reduce?
2. How does greedy placement compare with distance-priority and seeded random placement at the same budget?
3. How does the chosen population objective change locations and who receives modeled benefit?
4. Which origins have the largest backup-distance gap?
5. How does one facility loss change the same-budget allocation comparison?

Publish the exact input snapshot and methodology alongside any result. Revisit the original thesis questions with clear labels for what the new experiment does and does not answer.

## Present status

The source audit and this design are documentation deliverables. The session's executor failed to provision, and the user's complete local raw-to-final dataset is not mounted here. No app, Python test run, data reproduction, repaired pipeline or public release is claimed.

Implementation should continue in a working Codex/Claude Code session with the local repository and data available, using this file as the concrete scope and acceptance contract.
