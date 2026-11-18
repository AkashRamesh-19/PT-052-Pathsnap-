# PathSnap

> **Repository rollout: updated Part 1 only.** This commit contains schema-v3 data validation, WGS84 geometry, 20-feature extraction and ordered reference alignment, with nine standalone tests. The trained 12-feature model, HMM, API/UI and full-project verification are in the other teammates' assigned contributions and are not yet in this repository. Results below describe the verified local integrated project; full-system commands require Parts 2–4 to merge. Supply official data separately. Earlier preprocessing evidence is archived under `results/legacy_part1_v2/` and is not current schema-v3 output.

**Current verified pipeline:** schema v3 validates all 20 features and selects 12 using training-trip GroupKFold. The refitted model is in `artifacts/`; current results are in `results/retrained/`. Student-test ordered-alignment proxy accuracy is **87.20% classifier / 90.60% HMM as equal-weight means per trip**, and **85.59% / 88.89% pooled across GPS observations**. Only the mean per-trip HMM metric exceeds 90%. Retrain with `python -m pathsnap.train --samples-per-trip 256 --trials 6`. The earlier methodology, tables and PDF below describe historical experiments and are superseded by the current source/configuration and JSON results. No evaluator data or physical two-laptop verification was available.

Distributed GPS map matching and route reconstruction for UE24CS352A, PES University.

This repository implements data validation and exploration, candidate feature engineering, five supervised classifiers, trip-grouped cross-validation, Gaussian-process Bayesian optimisation, directed-network Viterbi decoding, a FastAPI service, a browser UI, Folium maps, and a generated technical report. The original consolidated hackathon requirements are preserved in [README (4).md](README%20(4).md).

**Important evidence boundary:** `.route` files contain a route sequence, not one label per GPS point. Current training and scoring use **sequence-aware alignment over ordered reference occurrences**, with ambiguity flags. These remain inferred proxy labels, not independently verified pointwise ground truth. Reference routes never enter prediction features or the inference decoder; optional student scoring opens references only after predictions finish, and evaluator inference never opens them.

## Quick start

Run from this folder using Python 3.12. The implementation was developed with Python 3.12.6 on Windows. No API keys, GPU or paid services are needed.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

If PowerShell activation is restricted, use `.\.venv\Scripts\python.exe` in place of `python`; changing execution policy is unnecessary. Linux/macOS activation: `source .venv/bin/activate`.

Train, compare, optimise and export reproducible artifacts:

```powershell
$env:OMP_NUM_THREADS = '2'
$env:OPENBLAS_NUM_THREADS = '2'
python -u -m pathsnap.experiment --samples-per-trip 160 --trials 8
```

The default uses up to 160 evenly spaced observations **per training trip** for local computational feasibility. Motion features use the original consecutive observations. Every student-test observation is evaluated. Use a higher `--samples-per-trip` to fit on more observations. Runs can take several minutes or longer, depending on hardware. Do not run multiple writers concurrently. Delete `data_cache/` and `results/inventory.json` before rerunning if raw development data changes; the cache assumes unchanged input files. The evaluator partition is never read by this command.

Generated artifacts are included when available, so launching the API does not require retraining. Load joblib artifacts only from trusted sources. Exact installed dependency versions are recorded in `results/environment.txt`.

## Two-laptop deployment

Both laptops need Python; only Laptop 1 needs the model dependencies, data and artifacts. Use the same source version on both laptops.

On **Laptop 1**, start the model API:

```powershell
python -m uvicorn pathsnap.api:app --host 0.0.0.0 --port 8000
```

On **Laptop 2**, serve the UI:

```powershell
python -m http.server 8080 --bind 0.0.0.0 --directory ui
```

Open `http://localhost:8080` on Laptop 2. Enter `http://<Laptop-1-LAN-IP>:8000` in the API address box and click **Connect to API**. Select a trip and click **Reconstruct route**. The map displays original GPS observations in blue and predicted road segments in green. It supports pan, zoom, layers and arc tooltips. The UI displays observation count, proxy accuracy, runtime and decoding resets. The API address is retained in that browser's local storage.

Both laptops must be on a network that permits connections to port 8000. Obtain Laptop 1's LAN IPv4 address with `ipconfig`. If needed, allow Python's API server through Windows Firewall on the trusted local network. `0.0.0.0` is a server bind address, not the address to enter in the browser. For a single-laptop demonstration, open `http://localhost:8000`; the API also serves the same UI.

The server defaults to the ten **student-test** trips for development. These are clearly identified in the UI and response. The reserved evaluator dataset was not supplied and is not replaced or fabricated.

For final evaluation only, install the untouched reserved ten trips under `evaluator_test/`, then start a fresh API process:

```powershell
$env:PATHSNAP_PARTITION = 'evaluator_test'
python -m uvicorn pathsnap.api:app --host 0.0.0.0 --port 8000
```

The UI automatically lists the server's partition. Missing evaluator data produces an empty list and an explicit not-ready message. Do not train, tune, select models or adjust thresholds using reserved trips or their results. Return to development with `$env:PATHSNAP_PARTITION = 'student_test'` and restart the server. Optional `PATHSNAP_UI_ORIGINS` accepts comma-separated allowed origins; the default permits local browser clients without credentials. This is a LAN demonstration service with no authentication; it is not configured as a public internet deployment.

## API contract

| Endpoint | Behaviour |
|---|---|
| `GET /health` | Service state, selected partition and model readiness |
| `GET /trips` | Partition, available trip IDs, expected count of ten and readiness |
| `POST /predict/{trip_id}` | Real model inference, Viterbi decoding, predicted arc IDs, reset indices, aggregate score and a Folium map |
| `GET /docs` | FastAPI interactive endpoint documentation |
| `GET /` | Browser interface for a single-laptop setup |

Example from another terminal:

```powershell
Invoke-RestMethod http://localhost:8000/health
Invoke-RestMethod http://localhost:8000/trips
$result = Invoke-RestMethod -Method Post http://localhost:8000/predict/00000000
$result | Select-Object trip_id, partition, observations, seconds, metrics
```

`00000000` is a supplied student-test trip; query `/trips` when using another partition. Requests log the partition, trip, observation count, duration, reset count and aggregate metrics in the **receiving API terminal**. Keep that terminal visible to evaluators. Missing models return HTTP 503; unknown trip IDs return 404; malformed raw data returns 422. No endpoint serves raw files or reference arc IDs. Evaluation opens a reference file only after prediction finishes and returns aggregate metrics. API maps omit reference geometry. The API does not bypass inference or return cached/hardcoded routes.

## Dataset contract and processing

```text
train/00000001/00000001.track   # longitude latitude timestamp_seconds
              00000001.nodes   # longitude latitude; row index = node ID
              00000001.arcs    # from_node to_node; row index = directed arc ID
              00000001.route   # traversed reference arc IDs, one per line
student_test/<trip-id>/...      # ten supplied development trips
evaluator_test/<trip-id>/...    # final evaluation only; not supplied
```

There must be exactly one `.track`, `.nodes` and `.arcs` file per trip; inference does not require `.route`. Raw files are whitespace-separated without headers. The loader validates shape, finite numbers, coordinate ranges, arc indices and strictly increasing timestamps. It rejects bad data explicitly rather than silently changing observations or road IDs. Ground-truth route indices are validated separately by the scoring/training layer.

The inventory records all 90 supplied development trips, observation counts, file sizes, timestamp gaps, coordinate extents, node/arc counts, weak connected components and isolated nodes. Invalid trips appear with an error. Train and student trip IDs must be disjoint. Raw data and caches are excluded from Git by `.gitignore`; obtain the official data separately when cloning.

Distances use a local equirectangular projection at the mean network latitude: longitude scale `111320*cos(latitude)` and latitude scale `111320` metres per degree. This is an approximation suitable for local networks, not a universal projection for continental routes. Coordinate order is always longitude, latitude internally and latitude, longitude for Folium.

The candidate generator finds the six nearest **segments**, not nearest nodes. A midpoint KD-tree gives an initial bound; all midpoints within that bound plus the largest segment half-length are checked by exact point-to-segment projection. This ensures long segments are not missed. If a network has fewer than six arcs, every arc is used. Distinct directed arcs preserve their own IDs even if they overlap geometrically.

The twenty input features and their units are listed in FEATURES.md; they cover geometric fit, directed alignment, motion, rolling speed context and road connectivity. Feature rationale: geometric fit, motion agreement and sampling reliability transfer across trip-specific networks, whereas raw arc IDs cannot be learned as global classes. Preprocessing is fit inside each training fold. A positive candidate label means equality with the nearest-reference-segment proxy; if no candidate matches the proxy, all candidates for that observation are negative.

## Models, experimental design and metrics

Five distinct models are trained:

| Model | Initial configuration and objective | Reason considered / limitation |
|---|---|---|
| Logistic regression | Scaled inputs; C=1; 400 iterations; logistic loss | Fast probability baseline; limited nonlinear interactions |
| Gaussian Naive Bayes | Scaled inputs; smoothing=1e-8; Gaussian likelihood | Very fast; feature independence is implausible for correlated geometry |
| Decision tree | Depth 10; leaf minimum 15; Gini impurity | Nonlinear rules; overfitting risk |
| Random forest | 80 trees; depth 14; leaf minimum 5; Gini impurity | Robust interactions; greater computation and memory |
| Histogram gradient boosting | 100 iterations; 15 leaves; L2=1; log loss | Flexible nonlinear probabilities; regularisation matters |

Seed: **352**. Three-fold `GroupKFold` keeps complete trips together using only the 80 training trips. The main model-family selection metric is mean student-test observation Top-1 proxy accuracy. Candidate Macro F1 is the grouped CV objective. Train-vs-CV F1 and fold dispersion expose possible overfitting. Results average each trip equally, so long trips do not dominate the final comparison.

Candidate accuracy, Macro precision/recall/F1, confusion matrices and log loss evaluate binary candidate relevance with each model's ordinary binary prediction/probability. Top-1 accuracy instead chooses the highest-scoring candidate for each observation. Candidate miss rate counts proxy reference segments absent from the candidates, and those observations remain incorrect in Top-1 accuracy. HMM accuracy uses the decoded segment per observation. Sequence exact match requires all proxy labels in a trip to match. Route continuity measures directed reachability between successive selected segments within the movement bound, not immediate endpoint adjacency. Timing includes feature generation and decoding, but excludes network loading, reference scoring and map rendering; baseline training time includes CV plus refit. Tuned training time is refit only; optimisation cost is recorded separately.

See `results/comparison.csv`, `results/comparison.json` and per-model trip JSON files for actual measurements. Comparison plots contain actual CV F1, student-test Top-1/F1/HMM accuracy and log loss; no artificial epoch curves are generated. The consolidated PDF contains separate model explanations and measured results.

## Bayesian optimisation

After comparing five models, a Gaussian-process surrogate with Matern and white-noise kernels searches the selected family. The first three trials initialise the surrogate; subsequent trials maximise expected improvement over 256 seeded proposals. Default: eight trials, objective: three-fold training-trip candidate Macro F1. No evaluator data is accessed.

| Family | Search space |
|---|---|
| Logistic regression | C: log-uniform 0.001–100 |
| GaussianNB | Smoothing: log-uniform 1e-12–1e-3 |
| Decision tree | Depth: 4–20; leaf minimum: 3–40 |
| Random forest | Depth: 6–22; leaf minimum: 2–20; fixed 80 trees |
| Histogram gradient boosting | Learning rate: log-uniform 0.01–0.2; leaves: 7–32 |

Trial scores, configurations, unit-space proposals and runtime are retained in `results/bayesian_trials.json`. The tuned model is refit and compared on the same student trips; final tuned-vs-untuned selection uses grouped CV F1. `artifacts/config.json` records the chosen configuration and feature order. Integer spaces use integer conversion, so exact upper endpoints have zero probability under continuous proposals.

## HMM / Viterbi

Hidden states are the candidate directed segments for each observation. Emissions are clipped binary classifier positive probabilities normalised across that observation's candidates. Transition log potential is `-abs(directed_road_distance - gps_displacement)/beta`. These are unnormalised HMM-style sequence potentials. Viterbi maximises summed log emission and transition scores and backtracks predecessor indices.

Directed shortest paths use bounded Dijkstra. Distances include projection fractions on the starting and ending segments; forward movement on the same arc uses its fraction difference directly. Impossible directed movements receive negative infinity. The search bound is `max(200, 55*gap_seconds + 100)` metres, capped at 10 km. Gaps over 60 seconds, or steps where no surviving transition is valid, start a new independent section and return a reset index. The map draws selected road segments separately so it does not create straight-line road connections over resets. The output is a segment for each GPS observation; it does not expand intervening shortest-path arcs into a full turn-by-turn route.

Beta is calibrated over 30, 60 and 120 metres on the permitted student-test partition, using HMM proxy accuracy then continuity. This is development tuning, not independent final evaluation. Results appear in `results/decoder_search.json`; final configuration results are in `results/final_metrics.json` and `results/final_trips.json`. Classifier-only vs decoder performance is recorded for all models; smoothing is not assumed to improve accuracy in every case.

## Repository and generated deliverables

| Path | Purpose |
|---|---|
| `pathsnap/data.py` | Raw validation, inventory, projection, exact candidates and network |
| `pathsnap/features.py` | Feature preparation, separate proxy labels, emissions |
| `pathsnap/experiment.py` | Five models, grouped CV, Bayesian search, export |
| `pathsnap/decoder.py` | Directed shortest paths, transition potentials, Viterbi and continuity |
| `pathsnap/maps.py` | Folium layers, GPS/network/reference/predicted routes |
| `pathsnap/api.py` | FastAPI model loading, inference and post-inference scoring |
| `ui/index.html` | Responsive two-laptop browser UI |
| `pathsnap/report.py` | Plots, comparison CSV and consolidated actual-results PDF |
| `tests/test_pipeline.py` | Candidate correctness, directionality, resets, label isolation and API configuration |
| `artifacts/model.joblib` | Final trusted trained model including learned preprocessing |
| `artifacts/config.json` | Features, candidate count, decoder and training provenance |
| `results/inventory.json` | Development EDA inventory |
| `results/comparison.*` | Measured model comparison |
| `results/*_trips.json` | Per-trip metrics and confusion matrices |
| `results/bayesian_trials.json` | Actual Bayesian search record |
| `results/representative_map.html` | Interactive development Folium map with reference layer |
| `results/*.png` | Actual EDA, comparisons and route overview |
| `reports/technical_report.pdf` | Consolidated generated technical report |

Regenerate the report after completed experiments with `python -m pathsnap.report`. Browser maps use Folium/Leaflet resources and OpenStreetMap tiles, which require internet access on the viewing laptop. Inference itself runs locally without external routing services. The report's static geographical plot is generated from real data; it is not represented as a Folium browser screenshot.

Run `python -m pathsnap.verify` to exercise every trip in the server-selected partition through FastAPI and retain an aggregate verification JSON. Use this only on student testing during development. `python -m pathsnap.calibrate` reruns the beta comparison with a fixed trained model using only student testing, then regenerates the report. Do not recalibrate after viewing reserved evaluator results.

With an API process running, `python -m pathsnap.verify --url http://127.0.0.1:8000` performs the same ten-trip checks over real HTTP, including the model and map responses.

## Submission readiness and remaining external work

Local verification completed: **6 automated tests passed**, UI returned HTTP 200 and passed JavaScript syntax checking, and **all 10 student-test trips (19,160 observations) passed real HTTP API requests** using the final model and decoder configuration. The retained aggregate record is `results/api_verification_student_test.json`. The PDF contains 19 pages of actual results, methodology and limitations. Browser rendering and the physical two-laptop demonstration have not been verified here.

The original requirements remain available in the source README. The implementation and actual local results address the software and development-experiment requirements. These external items cannot be invented or marked complete:

- Supply team names and identifiers, and the team's public GitHub URL. No remote repository or credentials were provided; nothing has been pushed.
- Add the reserved ten evaluator trips only for final evaluation and verify each through the API. Only the supplied student-test trips can be verified locally.
- Run the two-laptop demonstration and capture real UI and receiving-terminal screenshots.
- Open the generated Folium map and include a genuine screenshot/exported browser view in the final PDF. The generated report currently includes an actual static geographical plot.
- Review the generated report, add the above evidence and team details, and submit one consolidated PDF and one public repository.
- Disclose AI assistance according to the event policy. This project was implemented with Codex assistance; team members must understand, reproduce and defend all submitted code and results. AI-generated prose is not a substitute for independent experimental analysis.

Do not claim full final-submission readiness until these items are completed. Do not publish official raw datasets unless permitted by the course. No fabricated metrics, labels, screenshots or evaluator results are included.


<!-- measured-results:start -->
## Historical eight-feature results (superseded)

Actual development run: 80 training trips, 10 student-test trips, up to 160 training observations per trip, three-fold trip-grouped CV and 8 Bayesian trials. All accuracy values below use the documented geometric proxy labels. Values are means across student trips.

| Model | CV candidate Macro F1 | Student Top-1 | Student candidate Macro F1 | Log loss | HMM accuracy | HMM continuity | Inference seconds/trip |
|---|---:|---:|---:|---:|---:|---:|---:|
| LogisticRegression | 0.8350 | 0.7859 | 0.8520 | 0.2081 | 0.7779 | 0.9970 | 0.9865 |
| GaussianNB | 0.6542 | 0.7930 | 0.6620 | 0.7770 | 0.7886 | 0.9970 | 0.9784 |
| DecisionTree | 0.8673 | 0.8263 | 0.8903 | 0.2333 | 0.8106 | 0.9970 | 0.9997 |
| RandomForest | 0.8754 | 0.8308 | 0.8922 | 0.1468 | 0.8112 | 0.9970 | 1.1113 |
| HistGradientBoosting | 0.8783 | 0.8310 | 0.8932 | 0.1440 | 0.8126 | 0.9970 | 1.0341 |
| HistGradientBoosting_tuned | 0.8791 | 0.8317 | 0.8936 | 0.1443 | 0.8131 | 0.9970 | 1.1250 |

Final model: **HistGradientBoosting**, tuned configuration selected: **True**. Model-comparison decoding uses beta=60m; the final beta calibration is separate.

Final decoder beta: **30m**. Final classifier Top-1: **83.17%**; HMM proxy accuracy: **81.47%**; classifier continuity: **94.23%**; HMM continuity: **99.70%**.

These are development measurements, not reserved-evaluator results. See the PDF for model-specific configurations, limitations and interpretation.
<!-- measured-results:end -->
