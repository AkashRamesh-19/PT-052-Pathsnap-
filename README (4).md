# PathSnap — Distributed Map-Matching & Route Reconstruction

**Course:** UE24CS352A — Machine Learning
**Institution:** PES University, Department of Computer Science and Engineering, RR Campus
**Project:** PathSnap Hackathon

> This README consolidates the five supplied PathSnap documents: the Hackathon Guide, Dataset Description and Exploration, Models and Training Guidelines, Technical Report and Submission Guidelines, and Assessment and Academic Integrity Policy. It records the requirements in one place. Replace the clearly marked project-specific placeholders with the team's actual implementation details; do not invent results, commands, paths, or configurations.

---

## 1. Project Overview

GPS tracking systems can produce inaccurate location measurements because of urban buildings, signal interference, multipath effects, GPS dropouts, and low sampling frequencies. A vehicle travelling along a road may consequently appear to move onto a parallel street, jump across buildings, or deviate from its actual route.

**PathSnap** is a distributed machine-learning system intended to reconstruct realistic vehicle trajectories by matching noisy GPS observations to a road network. The end-to-end challenge covers raw-data understanding and processing, feature engineering, model experimentation, route reconstruction, and deployment.

The hackathon guide includes an illustrative Folium map: blue dots represent raw GPS observations and the green line represents the actual vehicle path (ground truth).

## 2. System Architecture

PathSnap uses a two-laptop deployment architecture:

| Machine | Component | Responsibilities |
|---|---|---|
| Laptop 1 | Model and API | Hosts the trained classifier, HMM/Viterbi implementation, road-network information, and server-side evaluation process. |
| Laptop 2 | User interface | Displays the ten evaluation-trip options, sends the selected trip to the API, and displays the returned route and results. |

### Evaluation request flow

1. Start the API on Laptop 1.
2. Start the UI on Laptop 2.
3. The evaluator opens the UI.
4. The UI displays the ten reserved evaluation trips.
5. The evaluator selects any one of the ten trips.
6. The UI sends the selected trip ID to the API.
7. The API processes the trip through the complete prediction pipeline.
8. The API calculates prediction accuracy using its internal reference data.
9. The UI displays the predicted route and evaluation results.

**All ten reserved trips must work**, not just one demonstration trip. Ground-truth labels must remain server-side and must never be exposed to the prediction model or UI. Hardcoded outputs and bypassing the API are prohibited. Evaluators must be shown the API logs on the laptop receiving inference requests.

## 3. Dataset Description

The supplied dataset contains GPS trajectory information and associated road-network information. A trajectory records observed vehicle movement over time; the road network represents the geographical structure against which GPS observations are matched. The raw package is organised by trip, with files describing GPS observations and relevant road-network information.

Do not assume every trip has identical file names, row counts, data types, or column meanings. Inspect the actual files and validate these details.

### 3.1 Raw file types

| Extension | Meaning | Structure described in the dataset guide |
|---|---|---|
| `.track` | GPS observations associated with a trajectory | Three whitespace-separated columns, no header: longitude, latitude, timestamp in seconds. Nominally 1 Hz; about 97% of steps are exactly one second apart. |
| `.nodes` | Geographical nodes in the road network | Two whitespace-separated columns, no header: longitude and latitude. The zero-based row index identifies the node. |
| `.arcs` | Connections between road-network nodes representing road segments | Two integer columns, no header: `from_node_id` and `to_node_id`. The zero-based row index serves as `arc_id`. |
| `.route` | Reference route information, where supplied | One integer column, no header: chronological sequence of ground-truth `arc_id` values traversed by the vehicle. |

**Important:** A row in a GPS file does not directly correspond to a row in a road-network file. Understand the relationships between GPS observations, road-network nodes, road segments, and reference routes before designing the processing pipeline.

### 3.2 Required initial dataset exploration

Inspect and record:

- Number of trips in each partition.
- Number of GPS observations per trip.
- File availability and file sizes.
- Data types and column names.
- Missing or invalid GPS observations.
- Timestamp ordering and irregular time gaps.
- Geographical extent of each trajectory.
- Number of road-network nodes and road segments.
- Road-network connectivity.
- Isolated nodes and disconnected components.
- Trips with unusually large GPS gaps or other anomalies.
- Availability and structure of reference-route information.

Recommended retained outputs:

1. Dataset inventory with trip counts, file counts, and observation counts.
2. Summary statistics for trajectory length, sampling intervals, and missing values.
3. Plots of GPS coordinate distributions and trajectory patterns.
4. Interactive Folium maps of selected GPS trajectories and the road network.
5. Tables or visualisations describing road-network size and connectivity.
6. A short record of data-quality issues, assumptions, and decisions.

At the end of exploration, the team should be able to explain the raw dataset organisation, each raw file's meaning, how the components relate, the main data-quality concerns, and the assumptions influencing the map-matching approach.

### 3.3 Geographical data considerations

- Longitude and latitude are angular coordinates.
- Ordinary Euclidean distance in degrees is generally not an appropriate physical-distance measure.
- Choose and justify an appropriate projected coordinate system or geodesic distance calculation.
- Check coordinate order carefully.
- Inspect coordinate plausibility before relying on spatial calculations.
- Consider GPS noise, low sampling rates, gaps, and irregular timestamps.
- Make geographical calculations appropriate for the dataset's scale and location.

### 3.4 Reference-route limitations

Investigate how supplied reference-route information is encoded and how it is associated with the road network. Treat it as reference information for the task, not automatically as independently verified, error-free physical ground truth. GPS observations may not lie exactly on the road network, and supplied routes may have limitations. Identify those limitations, explain assumptions, and avoid overstating the certainty of evaluation labels.

## 4. Dataset Partitions and Leakage Prevention

The complete dataset is divided as follows:

| Partition | Number of trips | Permitted use |
|---|---:|---|
| Training | 80 | Training, development, cross-validation, and model optimisation. |
| Student testing | 10 | Development-stage evaluation and model comparison. |
| Evaluator testing | 10 | Final evaluation only. |

Mandatory rules:

- No observations from the same trip may be distributed across different partitions.
- Evaluator-testing data must not be used for model training.
- Evaluator-testing labels must not be used for feature selection.
- Evaluator-testing results must not be used to choose algorithms or hyperparameters.
- Do not manually alter predictions based on known evaluation routes.
- Do not hardcode expected answers for evaluation trips.
- Fit learned preprocessing transformations using permitted training data only.
- Keep evaluator ground truth separate from the prediction pipeline.
- The evaluator-testing partition must remain untouched during development.

## 5. Expected Development Workflow

The hackathon guide specifies this overall sequence:

1. Understand the problem.
2. Explore and understand the raw dataset.
3. Develop an independent data-processing and feature-engineering approach.
4. Train and compare multiple machine-learning models.
5. Select the best-performing model based on experimental evidence.
6. Perform Bayesian hyperparameter optimisation.
7. Implement HMM- and Viterbi-based route reconstruction.
8. Export the trained model and required artefacts.
9. Develop the FastAPI backend.
10. Develop the evaluator-facing UI.
11. Integrate and test the complete system.
12. Prepare the final technical report.

Students are strongly encouraged to read all supporting documents before implementation or using AI tools. Relying only on AI-generated suggestions without understanding the supplied material may lead to incorrect implementation choices or incomplete solutions.

## 6. Data Processing and Feature Engineering

The repository must include the source code used to read, inspect, clean, and preprocess the raw dataset, along with the complete feature-extraction and feature-preparation code.

The team must document:

- How the data was preprocessed.
- What patterns were observed during exploration.
- Which features were extracted and why they were selected.
- Which dataset partition was used for each analysis.
- Any data-quality issues, assumptions, and processing decisions.

The supporting documents do not prescribe a fixed feature list or a single preprocessing algorithm. These decisions must be developed and justified independently from inspection of the actual dataset.

## 7. Model Experimentation and Comparison

### 7.1 Minimum model requirement

Train and evaluate **at least five distinct models** before selecting the final model. Experiment with models from multiple algorithm families where appropriate. Every model actually experimented with must be documented in the report. Directly selecting a model without meaningful experiments may result in deductions.

The objective is not merely to obtain a high accuracy score; the team must explain why a model is suitable for this problem and justify its selection with measured results and technical reasoning.

### 7.2 Suggested algorithm families

These are examples, not a fixed list:

| Algorithm family | Possible approaches | What to consider |
|---|---|---|
| Linear and probabilistic | Logistic Regression, Ridge Classifier, SGD Classifier, Gaussian Naive Bayes | Decision boundaries, assumptions, and probability estimates. |
| Instance-based | KNN, Distance-Weighted KNN, Radius Neighbours | Similarity, distance metrics, and scaling. |
| Tree-based | Decision Tree, Extra Trees | Nonlinear decision rules and overfitting. |
| Ensemble | Random Forest, Extra Trees Classifier, Bagging | Stability and generalisation. |
| Boosting | Gradient Boosting, HistGradientBoosting, XGBoost, LightGBM, CatBoost, AdaBoost | Sequential error correction and computational efficiency. |
| Kernel-based | Linear SVM, RBF SVM, Polynomial SVM | Kernel suitability and feature scaling. |
| Neural networks | Shallow MLP, Deep MLP, different hidden-layer configurations | Nonlinear representations, regularisation, and training stability. |
| Unsupervised and clustering | K-Means, Mini-Batch K-Means, Agglomerative Clustering, DBSCAN | Cluster interpretation and conversion into meaningful predictions. |

If an approach does not naturally produce supervised candidate probabilities, explain how it is adapted. Clustering identifiers must not be treated as ground-truth labels without an appropriate mapping or evaluation procedure.

### 7.3 Experimental design

The model experimentation process must:

1. Train at least five distinct models.
2. Record each model's configuration.
3. Use the 80 training trips for model fitting and cross-validation.
4. Use the ten student-testing trips for development-stage model comparisons.
5. Apply preprocessing consistently.
6. Avoid selecting models using training accuracy alone.
7. Record training time and inference time where practical.
8. Retain experiment logs, random seeds, configurations, and results.

### 7.4 Evaluation metrics

Report the following where applicable:

- Accuracy.
- Precision.
- Recall.
- F1-score.
- Confusion matrix.
- Log Loss.
- Observation-level Top-1 Accuracy.
- Candidate Miss Rate.
- Route Continuity.
- Sequence-level Performance.
- Training Time.
- Inference Time.

**Observation-level Top-1 Accuracy:** For each GPS observation, select the candidate with the highest predicted score and compare it with the corresponding reference road segment.

**Candidate Miss Rate:** Report cases where the reference road segment is absent from the generated candidate set.

Explain the primary metric used to select the final model and why it is appropriate for the route-matching objective.

### 7.5 Model comparison and selection

Compare at least five trained models. The comparison should identify:

- Model name and algorithm family.
- Important configuration choices.
- Training or cross-validation performance.
- Student-testing performance.
- Precision, recall, and F1-score.
- Log Loss, where applicable.
- Computational efficiency.
- Route Continuity.

Consider generalisation performance, stability across validation folds, probability quality, inference speed, model complexity, and compatibility with the sequential decoder. Do not select a model solely because it achieves high training accuracy.

### 7.6 Required model comparison table

Populate this table with actual experimental results. If a metric is not applicable to a model, enter `N/A` and explain why.

| Model | Algorithm family | Key configuration | CV Macro F1 | Student-test Top-1 Accuracy | Route continuity | Student-test Macro F1 | Log Loss | Inference time |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Model 1 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Model 2 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Model 3 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Model 4 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Model 5 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

### 7.7 Required questions for every model

Create a separate subsection for every model trained and answer:

**A. Model understanding**
1. Why was this algorithm considered suitable for the PathSnap problem?
2. What are its major strengths and limitations for this task?

**B. Training and configuration**
3. What were the important hyperparameters and their initial values?
4. What loss function or optimisation objective was used?

**C. Experimental results**
5. What was the model's performance on the student-testing dataset?
6. What were its observation-level Top-1 Accuracy, Macro F1-score, and Log Loss (where applicable) on the student-testing dataset?
7. Was overfitting or underfitting observed? What evidence supports the conclusion?

Do not provide only textbook descriptions. Connect each model's behaviour to the team's actual experiments, observations, and results.

### 7.8 Required model visualisations

Include visualisations that support model comparison and selection:

- Cross-validation Macro F1-score comparison across all models.
- Student-testing Top-1 Accuracy and Macro F1-score comparison.
- Log Loss comparison, wherever applicable.
- Confusion matrices may be included as diagnostic visualisations.
- Training and validation loss plots only for models that produce epoch-wise training histories.
- Folium maps of selected trips as part of EDA and wherever needed in the workflow.

Every graph must be generated from actual experimental results. Do not create artificial training curves for algorithms that do not produce training histories. Include comparative plots that justify model selection.

## 8. Bayesian Hyperparameter Optimisation

After the initial model comparison, select a promising model and perform hyperparameter optimisation using a **Bayesian search strategy**.

The optimisation process must:

1. Define a justified hyperparameter search space.
2. Specify the optimisation objective.
3. Use group-aware cross-validation on the 80 training trips.
4. Record optimisation trials and their results.
5. Identify the best parameter configuration.
6. Refit the selected model using the chosen configuration.
7. Evaluate the tuned model on the ten student-testing trips.
8. Compare tuned and untuned performance.
9. Report computational cost.

The report must state:

- Which hyperparameters were included in the search.
- What ranges or distributions were considered.
- What objective function was optimised.
- How many optimisation trials were performed.
- The best hyperparameter values obtained.
- Whether the tuned model outperformed the untuned model.

Compare the untuned and tuned models using the same metrics and data partitions:

- Cross-validation Macro F1-score.
- Student-testing Top-1 Accuracy.
- Student-testing Macro F1-score.
- Log Loss, where applicable.
- Inference Time.

**The evaluator-testing partition must not be used** to choose the search space, objective, hyperparameters, model, or thresholds.

## 9. HMM and Viterbi Route Reconstruction

Independent candidate predictions can produce unrealistic routes. For example, consecutive GPS observations may be assigned to road segments that are not connected in the road network. A Hidden Markov Model (HMM) uses sequential information and road-network connectivity to identify a more coherent route. The route reconstruction stage should use the selected model's outputs consistently.

The implementation must:

- Describe the hidden-state representation.
- Preserve candidate-to-road-segment correspondence.
- Explain emission and transition modelling choices.
- Explain the Viterbi recurrence or chosen decoding procedure.
- Handle missing candidates and invalid transitions.
- Handle GPS gaps appropriately.
- Return a predicted road segment for each relevant GPS observation.
- Compare classifier-only predictions with classifier-plus-HMM predictions.
- Report pointwise accuracy and sequence-level performance.
- Analyse route continuity.

Optimise decoder parameters using training and student-testing data only.

### 9.1 Required report questions

1. What are the hidden states in the implementation?
2. How are emission probabilities calculated?
3. How are transition probabilities calculated?
4. How are missing candidates or large GPS gaps handled?
5. What happens when consecutive GPS observations suggest disconnected road segments?
6. Did HMM decoding improve the final route? Support the answer with experimental results.

Include a comparison of:

- Classifier-only predictions.
- Classifier predictions followed by HMM/Viterbi decoding.

Discuss both improvements and cases where HMM decoding did not help.

## 10. Geographical Visualisation Using Folium

Folium-based geographical visualisation is **mandatory**. The report must include interactive geographical maps generated from the team's actual implementation.

At a minimum, visualise:

- Original GPS observations.
- Relevant road-network geometry.
- Reference route, where available.
- Predicted route.

Maps must:

1. Use distinguishable colours and markers for different route layers.
2. Include clear labels or a legend.
3. Have an appropriate geographical extent and zoom level.
4. Make it easy to compare the observed GPS trajectory with the predicted route.
5. Include layer controls where appropriate.

Submission requirements:

- Include at least one representative geographical map in the PDF as a screenshot or exported view.
- Commit the interactive Folium HTML map or the code required to generate it to GitHub.
- Include the map-generation code in the repository.
- Do not submit unrelated or manually fabricated maps.

## 11. API, UI, and Deployment

The project must include:

- A FastAPI backend that processes evaluation requests.
- Complete backend source code, including API endpoints and model-loading logic.
- A user interface that communicates with the API and displays evaluation results.
- Complete evaluator-interface source code.
- Configuration files, dependencies, and instructions required to run the system.
- Final model, preprocessing components, HMM configuration, and other required inference assets.

For the technical report, include:

- Screenshots of the working UI.
- Screenshots of successful API responses on the CMD/terminal of the receiving laptop.

The submitted source code must match the implementation demonstrated during evaluation. The API must process the selected trip through the complete prediction pipeline. The system must not expose evaluator ground-truth labels to the prediction model or UI, hardcode outputs, or bypass the API.

### 11.1 Project-specific run instructions — fill in from the actual repository

Do not leave this section as guessed commands. Add the exact commands that work on the evaluation machines.

```bash
# 1. Clone the public repository
# git clone <PUBLIC_REPOSITORY_URL>
# cd <REPOSITORY_DIRECTORY>

# 2. Create and activate the project's environment
# <ACTUAL_ENVIRONMENT_CREATION_AND_ACTIVATION_COMMANDS>

# 3. Install dependencies
# <ACTUAL_DEPENDENCY_INSTALL_COMMAND>

# 4. Start the API on Laptop 1
# <ACTUAL_FASTAPI_START_COMMAND>

# 5. Start the UI on Laptop 2
# <ACTUAL_UI_START_COMMAND>
```

**Repository URL:** `<ADD_PUBLIC_GITHUB_URL>`
**API address/port:** `<ADD_ACTUAL_API_ADDRESS_AND_PORT>`
**UI address/port:** `<ADD_ACTUAL_UI_ADDRESS_AND_PORT>`
**Environment setup:** `<ADD_ACTUAL_SETUP_DETAILS>`

## 12. GitHub Repository and Submission Deliverables

Each team must submit exactly two deliverables.

### Deliverable 1 — Public GitHub repository

The complete project source code must be committed and pushed to a public GitHub repository accessible to evaluators. The repository must be organised into clearly named folders and contain:

| Required component | What must be included |
|---|---|
| Data processing | Source code used to read, inspect, clean, and preprocess the raw dataset. |
| Feature engineering | Complete source code for feature extraction and preparation. |
| Model training | Training scripts/notebooks for every model experimented with. |
| Model evaluation | Code for model comparison, evaluation metrics, plots, and confusion matrices. |
| Hyperparameter optimisation | Bayesian optimisation code, search configurations, and experiment results. |
| HMM and Viterbi | Source code for route reconstruction and decoding. |
| API | Complete backend source code, including API endpoints and model-loading logic. |
| UI | Complete evaluator-interface source code. |
| Deployment | Configuration files, dependencies, and instructions required to run the system. |
| Visualisations | Folium map-generation code and other plotting scripts. |
| Supporting results | Relevant CSV files, experiment logs, configuration files, and result summaries. |

Additional repository rules:

- Commit all source code used for training, evaluation, UI, and API development.
- Do not submit only the final trained model; include the code used to produce it.
- Include saved model weights and required configuration files, subject to repository size constraints.
- Include a README with installation, execution, and deployment instructions.
- Organise the repository into clearly named folders.
- Ensure the final repository is accessible to evaluators.
- Never commit passwords, API keys, private credentials, or other secrets.

### Deliverable 2 — Consolidated technical report (PDF)

Submit one consolidated PDF report per team. It must follow the required structure in Section 13 below, include actual experimental results, tables, graphs, and geographical visualisations, and ensure every reported result can be verified using the submitted GitHub repository. Do not submit separate reports for individual models.

## 13. Required Technical Report Structure

The technical report should cover the following sections (the submission guide labels the report's points 2–7 as required report sections):

1. **Dataset Understanding and Exploratory Data Analysis**
   - Preprocessing approach.
   - Observed patterns and extracted features, with reasons for feature selection.
   - Missing-value analysis.
   - GPS trajectory plots and road-network visualisations.
   - At least one Folium map.
   - Explicitly identify the dataset partition used for each analysis.
2. **Model Experimentation and Comparison**
   - Consolidated comparison table for at least five models.
   - Separate subsection for every model actually experimented with.
   - Model suitability, strengths, limitations, hyperparameters, initial values, loss/objective, student-test results, Top-1 Accuracy, Macro F1, Log Loss where applicable, and overfitting/underfitting evidence.
   - Actual model-comparison plots.
3. **Bayesian Hyperparameter Optimisation**
   - Search parameters and ranges/distributions, objective, trial count, best values, computational cost, and tuned-versus-untuned comparison.
4. **HMM and Viterbi Route Reconstruction**
   - Hidden states, emissions, transitions, missing candidates/gaps, disconnected segments, Viterbi method, and classifier-only versus classifier-plus-decoder results.
5. **Geographical Visualisation Using Folium**
   - Original GPS observations, road geometry, reference route where available, and predicted route; representative map in the PDF and HTML/code in GitHub.
6. **API, UI, and Deployment**
   - Working UI screenshots and successful API response screenshots from the receiving laptop; implementation must match the demonstrated system.

Use actual results throughout. Do not fabricate metrics, charts, training histories, screenshots, or implementation details.

## 14. Assessment, Academic Integrity, and Independent Work

Assessment applies to source code, model implementations, model artefacts, configurations, experimental results, technical reports, and demonstrations. Evaluation considers correctness, completeness, technical understanding, experimental rigour, reproducibility, and the team's ability to explain its work. A high score on one metric does not automatically compensate for missing or invalid parts of the system.

### 14.1 Evidence of meaningful experimentation

Teams must:

- Train and compare at least five models before selecting a final model.
- Support model selection with recorded experimental results.
- Explain the strengths and limitations of the chosen approach.
- Demonstrate Bayesian hyperparameter optimisation.
- Retain the optimisation search record.
- Evaluate models and the decoder using permitted development data.

Selecting a model without meaningful experimentation may result in deductions.

### 14.2 Strict independent-work requirement

Each team must independently implement, train, evaluate, and select its own models. Sharing any of the following between teams is prohibited:

- Model files or trained weights.
- Parameter or hyperparameter configurations.
- Implementation code.
- Derived datasets.
- Experimental results.

Submitted architectures, source code, configurations, trained weights, model artefacts, and reported results may be compared across teams and sections. Unusually similar implementations or artefacts may be flagged for further review. Confirmed unauthorised sharing or copied work may result in severe consequences.

### 14.3 AI tool use and report authenticity

Reports and submitted written material may be checked for copied or AI-generated content. Teams remain responsible for the accuracy and originality of all submitted content.

The following are prohibited:

- Submitting AI-generated analysis as evidence of experiments that were not performed.
- Fabricating metrics, graphs, or code-execution results.
- Fabricating citations or implementation details.
- Copying another team's report or substantive content.
- Misrepresenting AI-generated work as independently conducted experimentation.
- Submitting technical explanations that the team cannot explain or defend.

AI tools must not replace actual experimentation, technical understanding, or independent analysis. Disclose AI assistance as required by the event's policies. Every team member must be able to explain the code, experiments, results, and conclusions in the report.

### 14.4 Viva and demonstration topics

Organisers may ask team members to explain:

- Dataset structure.
- Data preprocessing.
- Feature-engineering decisions.
- Model comparisons and model selection.
- Bayesian hyperparameter optimisation.
- HMM implementation and Viterbi decoding.
- API implementation.
- UI implementation.
- Experimental results.

Teams should be able to demonstrate that they understand and can reproduce their submitted work.

### 14.5 Similarity review

Organisers may compare:

- Source code and repository history.
- Model architectures and weights.
- Hyperparameter configurations.
- Experimental results.
- Technical reports, graphs, and tables.

Similarity may trigger a review or a request for further explanation. Standard algorithms and common approaches can naturally produce similarities; no single similarity measure should automatically be treated as proof of plagiarism.

Confirmed plagiarism, unauthorised collaboration, fabricated results, or violations of the AI-use policy may result in serious academic consequences, including mark deductions or disqualification, in accordance with the assessment and plagiarism policies.

## 15. Final Submission Checklist

### Dataset and data handling
- [ ] Dataset inventory and trip/observation counts recorded.
- [ ] Raw file structures and relationships understood and documented.
- [ ] Missing/invalid observations, timestamps, gaps, and anomalies analysed.
- [ ] Road-network size, connectivity, isolated nodes, and disconnected components analysed.
- [ ] Geographical distance method and coordinate handling justified.
- [ ] Dataset partition used for every analysis stated.
- [ ] No trip is split across partitions; evaluator data and labels remain protected.

### Models and experiments
- [ ] At least five distinct models trained and evaluated.
- [ ] Model families, configurations, and random seeds recorded.
- [ ] Preprocessing applied consistently.
- [ ] Training/CV and student-testing results recorded.
- [ ] Accuracy, precision, recall, F1, confusion matrix, Log Loss, Top-1 Accuracy, candidate miss rate, route continuity, sequence performance, and timing reported where applicable.
- [ ] Model-specific strengths, limitations, hyperparameters, objectives, results, and over/underfitting evidence documented.
- [ ] Model comparison table and actual comparison plots prepared.
- [ ] No artificial training curves or fabricated results.

### Optimisation and route reconstruction
- [ ] Bayesian search space, objective, trials, best parameters, and cost recorded.
- [ ] Group-aware cross-validation used on the 80 training trips.
- [ ] Tuned and untuned models compared on the same metrics and partitions.
- [ ] HMM hidden states, emissions, transitions, Viterbi procedure, and edge cases explained.
- [ ] Classifier-only and classifier-plus-HMM results compared.
- [ ] Decoder parameters optimised using training and student-testing data only.

### Maps, API, and UI
- [ ] Folium maps show GPS observations, road geometry, reference route where available, and predicted route.
- [ ] Maps have clear layers/colours/markers, labels or legend, appropriate zoom, and layer controls where appropriate.
- [ ] Representative map included in the PDF.
- [ ] Interactive Folium HTML or map-generation code committed to GitHub.
- [ ] API runs on Laptop 1 and UI runs on Laptop 2.
- [ ] UI displays all ten reserved evaluation trips.
- [ ] All ten trips work through the complete prediction pipeline.
- [ ] Ground truth remains server-side and hidden from the prediction model and UI.
- [ ] API logs shown on the receiving laptop; no hardcoded outputs or API bypass.
- [ ] Working UI and successful API response screenshots included in the report.
- [ ] Demonstrated implementation matches submitted source code.

### Repository and report
- [ ] Public GitHub repository contains all required source code, models/weights, configurations, dependencies, logs, and supporting results (subject to size constraints).
- [ ] Repository is clearly organised and evaluator-accessible.
- [ ] README includes verified installation, execution, and deployment instructions.
- [ ] No secrets or private credentials committed.
- [ ] One consolidated PDF report submitted per team.
- [ ] Every report result can be verified from the repository.
- [ ] Every team member can explain and reproduce the submitted work.
- [ ] AI assistance disclosed as required; no copied or fabricated content.

## 16. Team Implementation Details — Complete Before Submission

| Item | Team's actual details |
|---|---|
| Public GitHub repository | `<ADD_URL>` |
| Team members | `<ADD_NAMES/IDENTIFIERS>` |
| Dataset location / expected folder | `<ADD_ACTUAL_PATH>` |
| Preprocessing entry point | `<ADD_FILE_OR_NOTEBOOK>` |
| Feature engineering entry point | `<ADD_FILE_OR_NOTEBOOK>` |
| Model training entry point(s) | `<ADD_FILE_OR_NOTEBOOK>` |
| Model evaluation entry point | `<ADD_FILE_OR_NOTEBOOK>` |
| Bayesian optimisation entry point | `<ADD_FILE_OR_NOTEBOOK>` |
| HMM/Viterbi entry point | `<ADD_FILE>` |
| API entry point | `<ADD_FILE>` |
| UI entry point | `<ADD_FILE>` |
| Model and inference artefacts | `<ADD_ACTUAL_PATHS>` |
| Folium map output | `<ADD_ACTUAL_PATH>` |
| Technical report PDF | `<ADD_ACTUAL_PATH>` |

---

## Source Documents Consolidated

This README consolidates the following user-provided documents:

1. `Main Guidelines.pdf` — PathSnap: Distributed Map-Matching & Route Reconstruction, Hackathon Guide.
2. `Dataset Exploration.pdf` — PathSnap: Dataset Description and Exploration.
3. `Models and Training Guidelines.pdf` — PathSnap: Models and Training Guidelines.
4. `Deliverables.pdf` — PathSnap: Technical Report & Submission Guidelines.
5. `Assessment and Plagiarism Policy.pdf` — PathSnap: Assessment and Academic Integrity Policy.

Where the source documents describe requirements but do not prescribe an implementation, this README leaves the implementation decision to the team. Fill all placeholders with verified repository-specific information and actual experimental evidence before treating this README as the final project runbook.

## Implementation now available

The project implementation, installation commands, two-laptop deployment, model and decoder explanations, artifact paths, data limitations and submission status are documented in **[README.md](README.md)**. This file preserves the original requirements rather than claiming unchecked submission items are complete. Actual measured outputs are in `results/`, the trained model is in `artifacts/`, and the generated consolidated technical report is in `reports/technical_report.pdf` after experiments finish. Team identity, public GitHub URL, reserved evaluator verification and real demonstration screenshots still require the team's supplied information and final evaluation environment.
