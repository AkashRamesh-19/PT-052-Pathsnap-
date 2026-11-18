import numpy as np

FEATURE_SCHEMA_VERSION = 3
LEGACY_FEATURE_NAMES = ['distance_m', 'relative_distance_m', 'direction_cosine', 'direction_sine_abs', 'segment_length_m', 'fraction', 'speed_mps', 'gap_s']
FEATURE_NAMES = [
    'distance_m', 'relative_distance_m', 'direction_cosine', 'direction_sine_abs',
    'log_segment_length', 'fraction', 'speed_mps', 'gap_s',
    'displacement_m', 'acceleration_mps2', 'turn_angle_rad', 'speed_mean_5_mps',
    'speed_std_5_mps', 'distance_from_start_m', 'distance_to_end_m',
    'start_in_degree', 'end_out_degree', 'signed_cross_track_m',
    'along_road_displacement_m', 'candidate_distance_gap_m',
]

def prepare(trip, network, k=6, feature_names=None):
    """Twenty geometry/motion/topology features; reference routes are never used.

    feature_names explicitly selects a saved model's schema. The legacy name
    segment_length_m is an alias for log_segment_length, preserving old models.
    At observation zero, motion uses the first interval's true duration; speed
    subsequently uses backward differences. Heading uses up to three preceding
    steps to suppress GPS jitter. Acceleration/turn start at zero and reset at gaps.
    """
    ids, distance, fraction = network.candidates(trip.track, k)
    xy = network.project(trip.track[:, :2])
    motion = np.diff(xy, axis=0, prepend=xy[:1])
    motion[0] = motion[1]
    gap = np.diff(trip.track[:, 2], prepend=trip.track[0, 2]-1)
    gap[0] = gap[1]
    mag = np.linalg.norm(motion, axis=1)
    heading = xy-xy[np.maximum(np.arange(len(xy))-3,0)]
    heading[0] = motion[0]
    # Do not carry a heading across long discontinuities.
    recent_reset=0
    for i in range(1,len(xy)):
        if gap[i] > 60: recent_reset=i
        if i-recent_reset < 3: heading[i]=xy[i]-xy[max(recent_reset,i-3)]
    unit = heading / np.maximum(np.linalg.norm(heading,axis=1)[:,None], 1e-9)
    direction = network.vec[ids] / np.maximum(network.length[ids, None], 1e-9)
    cosine = np.sum(unit[:, None] * direction, axis=2)
    sine = np.abs(unit[:, None, 0] * direction[:, :, 1] - unit[:, None, 1] * direction[:, :, 0])
    shape = ids.shape
    speed = mag/gap
    acceleration = np.zeros(len(speed))
    acceleration[1:] = np.diff(speed)/gap[1:]
    turn = np.zeros(len(speed))
    turn[1:] = np.arctan2(
        unit[:-1, 0]*unit[1:, 1]-unit[:-1, 1]*unit[1:, 0],
        np.sum(unit[:-1]*unit[1:], axis=1))
    # One-sided rolling windows; restart context after a gap over 60 seconds.
    rolling_mean = np.empty(len(speed)); rolling_std = np.empty(len(speed))
    start = 0
    for i in range(len(speed)):
        if i and gap[i] > 60: start = i
        window = speed[max(start, i-4):i+1]
        rolling_mean[i], rolling_std[i] = window.mean(), window.std()
        if i and gap[i] > 60:
            acceleration[i] = 0.; turn[i] = 0.
    in_degree = np.bincount(trip.arcs[:, 1], minlength=len(trip.nodes))
    out_degree = np.bincount(trip.arcs[:, 0], minlength=len(trip.nodes))
    offset = xy[:, None, :] - network.a[ids]
    cross_track = direction[:, :, 0]*offset[:, :, 1]-direction[:, :, 1]*offset[:, :, 0]
    candidate_gap = distance[:, 1]-distance[:, 0] if shape[1] > 1 else np.zeros(len(speed))
    def broadcast(values): return np.broadcast_to(values[:, None], shape)
    x = np.stack([
        distance, distance-distance[:, :1], cosine, sine,
        np.log1p(network.length[ids]), fraction, broadcast(speed), broadcast(gap),
        broadcast(mag), broadcast(acceleration), broadcast(turn),
        broadcast(rolling_mean), broadcast(rolling_std),
        fraction*network.length[ids], (1-fraction)*network.length[ids],
        in_degree[trip.arcs[ids, 0]], out_degree[trip.arcs[ids, 1]],
        cross_track, np.sum(motion[:,None,:]*direction,axis=-1), broadcast(candidate_gap),
    ], axis=-1)
    if feature_names is not None:
        aliases = {'segment_length_m': 'log_segment_length'}
        selected = [aliases.get(name, name) for name in feature_names]
        if any(name not in FEATURE_NAMES for name in selected):
            raise ValueError('Unsupported saved-model feature schema; retrain the model')
        x = x[:, :, [FEATURE_NAMES.index(name) for name in selected]]
    return ids, fraction, x

def proxy_labels(trip, network, reference):
    from .alignment import align_reference
    return align_reference(trip, network, reference).labels

def probabilities(model, x):
    p = model.predict_proba(x.reshape(-1, x.shape[-1]))[:, list(model.classes_).index(1)]
    p = np.clip(p.reshape(x.shape[:2]), 1e-9, 1-1e-9)
    return p / p.sum(axis=1, keepdims=True)
