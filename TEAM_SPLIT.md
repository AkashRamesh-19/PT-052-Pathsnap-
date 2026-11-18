# PathSnap: four-person work allocation

This allocation assumes four contributors total. Names can be assigned to these roles without changing file ownership. The project was implemented locally with Codex assistance; splitting responsibility does not imply that four people independently authored the existing code. Each contributor must review, understand, test and explain their assigned part and disclose AI assistance as required.

| Contributor | Responsibility | Owned files | Evidence / acceptance check |
|---|---|---|---|
| 1 — proposed user role | Raw data validation, EDA inventory, geographical projection, candidate generation, features and proxy targets | `pathsnap/__init__.py`, `pathsnap/data.py`, `pathsnap/features.py`, data-specific tests; initial dependencies and ignore rules | Exact nearest candidates agree with brute force; invalid timestamps rejected; inference loader does not read labels; explain geometric proxy limitations |
| 2 | Five-model experiments, grouped validation, Bayesian tuning and model export | `pathsnap/experiment.py`, `artifacts/`, `results/comparison*`, `results/bayesian_trials.json`, per-model results, training-specific tests | Five measured models; whole-trip CV; recorded Bayesian trials; reproducible trusted model artifact; no evaluator access |
| 3 | Directed HMM/Viterbi, decoder calibration, metrics review, Folium maps and consolidated report | `pathsnap/decoder.py`, `pathsnap/calibrate.py`, `pathsnap/maps.py`, `pathsnap/report.py`, `reports/`, decoder/map/report result files and tests | Directed-path and reset tests; classifier-vs-HMM tradeoff; real-data maps; honest report and missing-evidence list |
| 4 | Backend, browser UI, two-laptop integration and end-to-end verification | `pathsnap/api.py`, `pathsnap/verify.py`, `ui/`, API tests, `results/api_verification_student_test.json` | All ten student trips over HTTP; no reference labels/geometry exposed; server logs; LAN demo and real screenshots |

## Shared-file rules

- Contributor 1 adds the initial `requirements.txt`, `.gitignore`, `README.md` and this allocation. Later contributors extend the README and dependency list in small reviewed changes.
- Split the existing mixed `tests/test_pipeline.py` by responsibility when integrating; do not have multiple contributors overwrite the same test file.
- Contributor 2 owns the experiment orchestrator. Contributor 3 coordinates changes where it invokes decoder calibration, map generation and reports.
- Each contributor owns their generated result paths. Run training/calibration sequentially to prevent concurrent artifact writers.
- Keep official raw data, caches, virtual environments and logs out of Git. Do not publish evaluator data. Retain reproducible aggregate JSON/CSV experiment records.

## Integration order and interfaces

1. Merge contributor 1's foundation.
2. Contributors 2 and 3 work against the foundation in separate branches. Decoder unit tests can run once roles 1 and 3 merge. Full experiments and report generation require all four contributions: the experiment script imports decoder/report components, and the report uses the API inference helper.
3. Merge contributor 4 after model export and decoder/map interfaces are present. Contributor 4 can build the UI before then, but must not claim mocked responses as real inference.
4. Run unit checks and all ten student-trip HTTP requests on the integrated branch. Capture the real two-laptop demonstration and final screenshots.

Stable interfaces:

- `read_trip(folder) -> Trip(id, track, nodes, arcs)` never opens `.route`.
- `read_reference(folder, arc_count)` belongs only to training/scoring.
- `Network(trip)` exposes metre-based geometry and directed adjacency.
- `prepare(trip, network, k=6) -> (candidate_ids, fractions, features)`; features are observation × candidate × feature.
- `proxy_labels(trip, network, reference)` produces geometric proxy targets, not verified observation ground truth.
- `probabilities(model, features)` normalises binary positive probabilities within each candidate set.
- `Decoder(network, beta, max_speed, gap_reset).decode(...) -> (arc_ids, reset_indices)`.
- Backend response fields and deployment instructions are documented in the integrated project's README.

## Git workflow

Use a contribution branch, such as `feature/data-features`, and push only the files owned by that role. Clone and inspect the actual remote before copying files; preserve existing repository files and history. Do not force-push or stage the entire local completed project as one person's contribution. Open a pull request or follow the team's agreed merge process.

The isolated `team_handoff/data-features/` directory is a proposed contributor-1 package. It contains no model, API, decoder, raw data or generated training results. A local contribution branch is prepared there; this is not evidence of a successful remote push. Integrate onto the actual remote history after its URL is supplied.
