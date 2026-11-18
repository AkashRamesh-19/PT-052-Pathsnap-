import numpy as np

FEATURE_NAMES = ['distance_m', 'relative_distance_m', 'direction_cosine', 'direction_sine_abs', 'segment_length_m', 'fraction', 'speed_mps', 'gap_s']

def prepare(trip, network, k=6):
    ids, distance, fraction = network.candidates(trip.track, k)
    xy = network.project(trip.track[:, :2])
    motion = np.diff(xy, axis=0, prepend=xy[:1])
    motion[0] = motion[1]
    gap = np.diff(trip.track[:, 2], prepend=trip.track[0, 2]-1)
    mag = np.linalg.norm(motion, axis=1)
    unit = motion / np.maximum(mag[:, None], 1e-9)
    direction = network.vec[ids] / np.maximum(network.length[ids, None], 1e-9)
    cosine = np.sum(unit[:, None] * direction, axis=2)
    sine = np.abs(unit[:, None, 0] * direction[:, :, 1] - unit[:, None, 1] * direction[:, :, 0])
    shape = ids.shape
    x = np.stack([distance, distance-distance[:, :1], cosine, sine, np.log1p(network.length[ids]), fraction, np.broadcast_to((mag/gap)[:, None], shape), np.broadcast_to(gap[:, None], shape)], axis=-1)
    return ids, fraction, x

def proxy_labels(trip, network, reference):
    # Route files are sequences, not observation labels. This geometric proxy
    # cannot resolve repeated traversals or direction ties; report that explicitly.
    return network.candidates(trip.track, k=1, subset=reference)[0][:, 0]

def probabilities(model, x):
    p = model.predict_proba(x.reshape(-1, x.shape[-1]))[:, list(model.classes_).index(1)]
    p = np.clip(p.reshape(x.shape[:2]), 1e-9, 1-1e-9)
    return p / p.sum(axis=1, keepdims=True)
