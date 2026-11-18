# Phase 1: twenty input features (schema v3)

Each GPS observation has up to six candidate directed road segments. Every candidate gets these **20 numeric features**. Observation index, timestamp and arc ID identify rows and are not counted as extra features. No reference-route labels or learned transformations are used to extract them.

| # | Feature | Meaning and purpose |
|---:|---|---|
| 1 | `distance_m` | Exact distance to the candidate segment; geometric GPS fit. |
| 2 | `relative_distance_m` | Distance minus the closest candidate distance; relative fit. |
| 3 | `direction_cosine` | GPS-motion versus directed-road alignment; negative means opposite travel. |
| 4 | `direction_sine_abs` | Absolute cross-direction alignment; lateral motion. |
| 5 | `log_segment_length` | Natural log of 1 + segment length in metres; compresses road-length scale. |
| 6 | `fraction` | Clipped projection position along the segment, from 0 to 1. |
| 7 | `speed_mps` | GPS displacement divided by actual time gap. |
| 8 | `gap_s` | Actual timestamp interval; sampling reliability. |
| 9 | `displacement_m` | GPS movement distance in the interval; distinguishes short and long steps at similar speed. |
| 10 | `acceleration_mps2` | Change in GPS speed divided by time gap; movement dynamics. |
| 11 | `turn_angle_rad` | Signed change in GPS heading in [-pi, pi]; turning behaviour. |
| 12 | `speed_mean_5_mps` | Mean speed over the current and up to four preceding observations; local motion context. |
| 13 | `speed_std_5_mps` | Population standard deviation over that window; motion variability. |
| 14 | `distance_from_start_m` | Projection fraction times segment length; progress from the directed start. |
| 15 | `distance_to_end_m` | Remaining segment length from the projected point; proximity to the directed end. |
| 16 | `start_in_degree` | Number of directed arcs entering the candidate's start node; incoming connectivity. |
| 17 | `end_out_degree` | Number of directed arcs leaving the candidate's end node; outgoing choices. |
| 18 | `signed_cross_track_m` | Signed perpendicular offset from the candidate's infinite directed line; side of the road, positive to its left. Unlike segment distance, this does not include endpoint overshoot. |
| 19 | `along_road_displacement_m` | GPS movement projected onto candidate direction; signed forward progress. |
| 20 | `candidate_distance_gap_m` | Second-nearest minus nearest candidate distance, shared across candidates; geometric ambiguity. Zero with only one available candidate (no comparison). |

Geometry uses the documented local metre projection. GPS-derived speeds/accelerations are noisy observations, not verified vehicle dynamics. The first motion estimate uses the first two observations (offline processing); later estimates use backward differences. Acceleration and turning initialise to zero. A stationary or zero-length road direction has zero vector alignment. Rolling windows use only current/previous speeds and restart after gaps over 60 seconds; acceleration/turn also reset at those gaps. Degree counts count directed arcs, including parallel connections.

Several features are related by geometry. Their inclusion provides interpretable representations for different model families; having 20 features does not prove they improve performance. Feature utility and redundancy need training-only validation/ablation later.

## Evidence and model compatibility

Run `python -m pathsnap.phase1` from the project root with the official training data present. The terminal prints all 20 feature values for a real candidate and saves `phase1_demo/preprocessed_candidates.csv` with 22 columns: 2 row identifiers + 20 inputs. This demo never trains a model or accesses evaluator data.

The current model uses 12 features selected from the validated 20 using training-trip grouped validation. Geometry uses a WGS84 local azimuthal-equidistant projection; heading uses up to three previous steps to suppress jitter. `artifacts/feature_schema.json` records the selected names and hash, which the API validates against the saved model. Current results are in `results/retrained/`; the older root-level comparison and PDF remain historical eight-feature evidence. Schema-v3 caches prevent reuse of old preprocessing. Legacy `segment_length_m` is an extraction alias for `log_segment_length`, but the API rejects old model schemas.

To train and measure the new schema, run the full experiment command after all teammates' modules are integrated. Do not reuse old accuracy figures as evidence of a twenty-feature model.
