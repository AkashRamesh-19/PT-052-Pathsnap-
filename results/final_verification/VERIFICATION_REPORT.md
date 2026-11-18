# Final PathSnap execution verification

Verdict: **READY WITH MANUAL CHECKS**. Software executed on one Windows machine; this is not proof of physical two-laptop or reserved-evaluator execution.

| Component | Actually executed? | Evidence | Expected result | Actual result | Remaining issue |
|---|---|---|---|---|---|
| ML artifacts / features | PASS | training_policy.json; artifacts/config.json; unit_tests.txt | Matching schema and >=12 used inputs | 12 inputs and split features; model reload predictions identical | Proxy labels are inferred, not verified truth |
| Five-model experiments / Bayesian selection | PASS | results/retrained/model_comparison.json; bayesian_trials.json; training_policy.json | >=5 models; genuine training-only optimisation | Five models; 6 trials; 3 GP/EI proposals replay exactly; best parameters match final model | Student data has historical development use; not a pristine holdout |
| Data partitions / grouped CV | PASS | training_policy.json | 80 train / 10 student, disjoint trip folds | Fresh parsing of training data; disjoint 3-fold trip groups | Reserved data absent |
| HMM + Viterbi | PASS | fresh_metrics.json; predictions/*.npz | Real classifier probabilities, directed paths, gap handling | All 19,160 observations recalculated; 16 decoder settings on nine training OOF trips | HMM does not improve every trip |
| FastAPI backend | PASS | http.json; http_requests.txt; backend_log_excerpt.txt | Live model and real JSON predictions | 10/10 student requests match independent predictions; 404/405 errors and CORS checked | Student trips are NOT the reserved trips |
| Frontend + Folium | PASS | browser_e2e.json; ui_*.png; map_*.png | Real browser POSTs, GPS/roads/prediction, legend, bounds | Two trips, real loaded tiles, GPS bounds, mobile layout and invalid-URL recovery | Internet needed for external map resources |
| End-to-end pipeline | PASS | browser_e2e.json; http.json; fresh_metrics.json | UI selection through dynamic ML/HMM to map | No mocked API; model output matches fresh reference-free inference | None found in tested development flow |
| Two-laptop integration | NOT VERIFIED | Both listeners bind 0.0.0.0:8000/8080; CORS HTTP preflight passes | Successful requests from a second physical laptop | Separate API/frontend processes verified on one machine | Test actual LAN/firewall/device connection |
| Reserved evaluator trips | NOT VERIFIED | No evaluator_test folder; inference-leakage unit tests pass | Run actual ten reserved trips without opening hidden references | No reserved data or labels accessed | Obtain and run only under final-evaluation rules |
| Current required plots | PASS | results/retrained/cv_macro_f1.png; student_accuracy_f1.png; student_log_loss.png | Actual five-model comparison plots | Generated from current recorded JSON metrics | Include these in the team's report |
| GitHub / consolidated report | FAIL | remote_files.txt; historical PDF noted in README | Current full source, artifacts and current report | Remote main contained Part 1 only when fetched; old PDF is not a current report | Team must push assigned ZIP payloads, add details and update report; nothing published here |

## Confirmed results

- Final model: Bayesian-tuned HistGradientBoosting; 12 actual inputs: distance_m, relative_distance_m, direction_cosine, direction_sine_abs, log_segment_length, fraction, speed_mps, gap_s, displacement_m, acceleration_mps2, turn_angle_rad, speed_mean_5_mps.
- Classifier: **87.20% mean per trip / 85.59% pooled**.
- HMM: **90.60% mean per trip / 88.89% pooled**. Only the trip-mean HMM result exceeds 90%.
- Denominators: 10 equally weighted trips for means; 19,160 GPS observations for pooled scores. Candidate misses stay in denominators. Candidate miss rate: 0.26%; HMM continuity: 99.97%; sequence exact-match: 0.00%. Continuity excludes configured long gaps.
- Observation Macro F1: 0.736068; observation log loss: 0.454708. Candidate-level precision/recall/F1, accuracy and confusion matrices are retained per trip in fresh_metrics.json.
- Bayesian parameters: learning_rate=0.031556125312966504, max_leaf_nodes=29, L2=4.959462605261326, iterations=120, early_stopping=False, seed=352.
- Automated tests: **23 passed, 0 failed**. ML weights unchanged during this audit. No training, student-driven parameter changes, evaluator-label access or GitHub publication performed.
- Historical development used student data in earlier experiments. Current final selection/tuning is training-only; do not describe student scores as a pristine independent holdout.

## Actual command and response evidence

```text
python -m pathsnap.audit
python -m pathsnap.audit_http
python -m unittest discover -s tests -v
python tests/browser_e2e.py --trips 2 --screenshots --output results/final_verification
GET /health -> 200; model_ready=true, feature_count=12
POST /predict/00000000 -> 200; observations=2503
POST /predict/not-a-trip -> 404
POST /predict/99999999 -> 404
GET /predict/00000000 -> 405
Cross-origin POST preflight -> 200
```

Real receiving-server log excerpt: backend_log_excerpt.txt. Full HTTP summaries: http_requests.txt. First predicted arc IDs, durations and per-trip response summaries: http.json. Independently saved states, probabilities and aligned labels: predictions/*.npz. No static route output was substituted.

Screenshots (real browser, individual trip scores are not global accuracy):
- results/final_verification/ui_00000000.png
- results/final_verification/map_00000000.png
- results/final_verification/ui_00000007.png
- results/final_verification/map_00000007.png
- results/final_verification/mobile_ui.png

## Commands for the professor's demo

From the extracted full project root (or the integrated team repository), install matching dependencies once:

```powershell
python -m pip install -r requirements-lock.txt
python -m unittest discover -s tests -v
```

Laptop 1 / API terminal:

```powershell
$env:PATHSNAP_PARTITION='student_test'
$env:PATHSNAP_SCORE_STUDENT='1'
python -m uvicorn pathsnap.api:app --host 0.0.0.0 --port 8000
```

Laptop 2 / frontend terminal:

```powershell
python -m http.server 8080 --bind 0.0.0.0 --directory ui
```

Open http://localhost:8080 on Laptop 2 and set the API address to http://<Laptop-1-LAN-IP>:8000. Keep the receiving API terminal visible. For a one-machine demo use http://127.0.0.1:8000 as the API address. Reserved inference uses PATHSNAP_PARTITION=evaluator_test only once the untouched ten trips are provided; reference files are not read in that partition even if student scoring was enabled.

## Changes and remaining manual work

This audit added verification scripts, independent prediction records, real screenshots, current comparison plots, explicit accuracy weighting and a dependency lock. It enhanced browser checks for a second trip, errors, actual GPS map bounds and loaded tiles. No ML predictor or decoder parameters were changed and no retraining/PDF/publication occurred.

Priority 1: Test the physical LAN and supply the reserved trips under the competition rules. Priority 2: Push all four assigned source contributions; the remote currently lacks parts 2-4. Priority 3: Update the consolidated report with current metrics, plots and genuine screenshots; add team names, repository URL, receiving-terminal screenshot and required AI-assistance disclosure. Each team member must understand and defend their assigned work.
