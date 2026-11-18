# PathSnap — GPS Map Matching and Route Reconstruction

**Course:** UE24CS352A — Machine Learning  
**Project:** PathSnap Hackathon  
**Institution:** PES University, Bengaluru

## 1. Project Overview

PathSnap is a GPS map-matching and route-reconstruction system that converts noisy GPS observations into realistic vehicle trajectories by matching them to a directed road network.

GPS observations may contain noise, inaccurate coordinates, signal interruptions and irregular sampling. PathSnap uses candidate road-segment generation, feature engineering, machine-learning classification and HMM/Viterbi sequence decoding to reconstruct likely vehicle routes.

## 2. System Architecture

The project follows this processing pipeline:

1. Load and validate GPS observations and road-network data.
2. Generate nearby road-segment candidates for each GPS observation.
3. Extract geometric, directional and motion-related features.
4. Train and compare five machine-learning classifiers.
5. Apply Bayesian hyperparameter optimisation to the selected classifier.
6. Use classifier probabilities as emission evidence for HMM-style Viterbi decoding.
7. Reconstruct connected road paths.
8. Serve predictions through FastAPI and display them using a browser-based map interface.

## 3. Dataset

The dataset contains four main file types:

| File | Description |
|---|---|
| `.track` | GPS longitude, latitude and timestamps |
| `.nodes` | Road-network node coordinates |
| `.arcs` | Directed road connections |
| `.route` | Ordered reference road-arc sequence |

The supplied development dataset contains 80 training trips and 10 student-testing trips. Ten reserved evaluator trips are intended for authorised final evaluation.

Reference routes are used for training-label alignment and permitted post-inference scoring. They are not used as input features during prediction. The reported accuracy values are based on inferred ordered-reference alignment proxy labels, not independently verified GPS ground truth.

Raw datasets are not included in the repository. Obtain the official data separately and follow the course data-sharing rules.

## 4. Feature Engineering

The implementation extracts 20 candidate features and uses a selected set of 12 features in the final model.

The selected features represent:

- GPS-to-road distance and relative distance
- Directional agreement
- Road-segment length and position
- Vehicle speed and time gaps
- GPS displacement and acceleration
- Turn angle
- Rolling speed context

The feature schema and exact feature order are stored in the `artifacts/` directory.

## 5. Machine-Learning Models

Five classifiers were compared:

1. Logistic Regression
2. Gaussian Naive Bayes
3. Decision Tree
4. Random Forest
5. HistGradientBoosting

Model evaluation uses trip-grouped cross-validation to keep observations from the same trip together.

HistGradientBoosting was selected and tuned using Gaussian-process Bayesian optimisation. The final trained model and configuration are stored in `artifacts/`.

## 6. HMM and Viterbi Decoding

The selected classifier produces probabilities for candidate road segments. These probabilities provide emission evidence to the sequence decoder.

The decoder combines classifier evidence with directed road-network transition information and GPS displacement. The Viterbi algorithm identifies the most likely sequence of road segments.

Route reconstruction then connects selected segments through valid directed road-network paths where possible.

## 7. Experimental Results

The current student-test results are:

| Metric | Classifier | HMM/Viterbi |
|---|---:|---:|
| Mean per-trip Top-1 accuracy | 87.20% | 90.60% |
| Pooled observation accuracy | 85.59% | 88.89% |

Mean HMM continuity is approximately 99.97%.

These are ordered-reference alignment proxy measurements. The 90.60% value is an equal-weight average across trips; pooled accuracy is 88.89%. The 95% engineering target was not achieved.

See `results/retrained/` and `results/final_verification/` for the recorded experiment and verification evidence.

## 8. Installation

Use Python 3.12.

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the pinned dependencies:

```powershell
python -m pip install -r requirements-lock.txt
```

## 9. Run the Application

Start the API in the project root:

```powershell
$env:PATHSNAP_PARTITION = 'student_test'
python -m uvicorn pathsnap.api:app --host 0.0.0.0 --port 8000
```

Open the API documentation:

`http://localhost:8000/docs`

For a separate browser UI, open another terminal:

```powershell
python -m http.server 8080 --bind 0.0.0.0 --directory ui
```

Open:

`http://localhost:8080`

Enter the API address in the interface, connect, select a trip and request route reconstruction.

For a two-laptop demonstration, run the API on Laptop 1 and the UI on Laptop 2. Enter Laptop 1's LAN IPv4 address in the UI on Laptop 2. Both devices must be on a network that permits communication between them.

## 10. Reproduce Training and Tests

Train and compare the models:

```powershell
python -m pathsnap.train --samples-per-trip 256 --trials 6
```

Run the automated tests:

```powershell
python -m unittest discover -s tests -v
```

Run the training-policy audit:

```powershell
python -m pathsnap.audit
```

Run the HTTP verification:

```powershell
python -m pathsnap.audit_http
```

Run browser end-to-end verification:

```powershell
python tests/browser_e2e.py --trips 2 --screenshots --output results/final_verification
```

Use only the student-testing partition for development and tuning. The reserved evaluator partition must remain isolated until authorised final evaluation.

## 11. Repository Structure

| Directory / file | Purpose |
|---|---|
| `pathsnap/data.py` | Dataset loading and validation |
| `pathsnap/features.py` | Candidate generation and feature extraction |
| `pathsnap/alignment.py` | Ordered reference-route alignment |
| `pathsnap/train.py` | Model training, comparison and optimisation |
| `pathsnap/experiment.py` | Experiment workflow |
| `pathsnap/decoder.py` | HMM-style transition scoring and Viterbi decoding |
| `pathsnap/reconstruct.py` | Directed road-path reconstruction |
| `pathsnap/api.py` | FastAPI backend |
| `pathsnap/maps.py` | Folium map generation |
| `ui/index.html` | Browser interface |
| `artifacts/` | Final model and configuration |
| `results/retrained/` | Current experiment results |
| `results/final_verification/` | Verification evidence |
| `tests/` | Automated tests |

## 12. Limitations

- Reference labels are inferred from ordered route alignment and are not independently verified GPS ground truth.
- Candidate generation may fail to include the correct road segment.
- GPS noise and overlapping opposite-direction roads can create ambiguity.
- HMM decoding does not improve accuracy on every trip.
- Exact full-sequence matching was not achieved.
- Reserved evaluator-trip results are not available.
- Physical two-laptop deployment must be verified separately.
- The application is intended for a controlled LAN demonstration, not production deployment.

## 13. Submission and Reproducibility

The repository contains the current application source, trained model, configuration, experiment records and verification evidence.

The final technical report should be consistent with the committed model, source code and results.

All team members should understand the implementation and be able to explain and defend their contributions. AI assistance must be disclosed according to the course and hackathon policies.
