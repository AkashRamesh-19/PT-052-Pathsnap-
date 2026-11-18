# PathSnap: four-person work allocation

**Current handoff:** use `team_packages/CURRENT_VERIFIED_2026-11-18/`. Earlier ZIPs and `team_handoff/` snapshots are historical. Current artifacts use schema v3 and 12 selected inputs from 20 validated features. Exact non-overlapping ownership is recorded in each current ZIP's `FILE_HASHES.json` and `HOW_TO_PUSH.md`.

This allocation assumes four contributors total. Names can be assigned to these roles without changing file ownership. The project was implemented locally with Codex assistance; splitting responsibility does not imply that four people independently authored the existing code. Each contributor must review, understand, test and explain their assigned part and disclose AI assistance as required.

| Contributor | Responsibility | Owned files | Evidence / acceptance check |
|---|---|---|---|
| 1 — your part | Raw validation, WGS84 geometry, EDA, candidates, features and ordered alignment | `data.py`, `features.py`, `alignment.py`, `phase1.py`, `baseline.py`, package initializer; data/alignment tests; dependencies, shared docs and ignore rules | Validated 20-feature extractor; ordered inferred labels with uncertainty; reference-free feature extraction |
| 2 | Five models, grouped selection, Bayesian tuning and final export | `train.py`, compatibility `experiment.py`, the three current root-level artifact files, current model/ablation/candidate result records, comparison plots and training-policy tests | Five measured models; training-only trip groups; six replayed Bayesian trials; final 12-feature model |
| 3 | HMM/Viterbi, path reconstruction, calibration and map generation | `decoder.py`, `reconstruct.py`, `calibrate.py`, `maps.py`, `diagnose.py`, legacy guarded `report.py`, decoder/map results and reconstruction tests | Directed paths and gaps; 16 training-only decoder settings; genuine reconstructed routes; historical PDF excluded |
| 4 | API, UI and complete verification | `api.py`, `verify.py`, `audit.py`, `audit_http.py`, `ui/`, browser/API/integration tests, current verification report, prediction records, screenshots and package utility | Ten student HTTP requests; two real Chrome trips; no hidden reference inputs; mobile/error/bounds checks |

## Shared-file rules

- Contributor 1 adds the initial `requirements.txt`, `.gitignore`, `README.md` and this allocation. Later contributors extend the README and dependency list in small reviewed changes.
- `tests/test_pipeline.py` belongs to contributor 4 as a shared integration suite. Other contributors own the separate test files listed in their ZIP manifests; do not overwrite one another's tests.
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
