"""Reproducible five-model comparison, grouped CV, GP Bayesian search and reports."""
import argparse
import csv
import json
import time
from pathlib import Path
import joblib
import numpy as np
from scipy.stats import norm
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, log_loss
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from .data import ROOT, read_trip, read_reference, Network, inventory
from .features import prepare, proxy_labels, probabilities, FEATURE_NAMES, FEATURE_SCHEMA_VERSION, LEGACY_FEATURE_NAMES
from .decoder import Decoder

SEED = 352
OUT = ROOT/'results'
ART = ROOT/'artifacts'

def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')

def models():
    return {
        'LogisticRegression': make_pipeline(StandardScaler(), LogisticRegression(C=1., max_iter=400, random_state=SEED)),
        'GaussianNB': make_pipeline(StandardScaler(), GaussianNB(var_smoothing=1e-8)),
        'DecisionTree': DecisionTreeClassifier(max_depth=10, min_samples_leaf=15, random_state=SEED),
        'RandomForest': RandomForestClassifier(n_estimators=80, max_depth=14, min_samples_leaf=5, n_jobs=2, random_state=SEED),
        'HistGradientBoosting': HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=15, l2_regularization=1., early_stopping=False, random_state=SEED),
    }

def tuned_model(name, z):
    if name == 'LogisticRegression':
        return make_pipeline(StandardScaler(), LogisticRegression(C=10**(-3+5*z[0]), max_iter=500, random_state=SEED))
    if name == 'GaussianNB':
        return make_pipeline(StandardScaler(), GaussianNB(var_smoothing=10**(-12+9*z[0])))
    if name == 'DecisionTree':
        return DecisionTreeClassifier(max_depth=int(4+16*z[0]), min_samples_leaf=int(3+37*z[1]), random_state=SEED)
    if name == 'RandomForest':
        return RandomForestClassifier(n_estimators=80, max_depth=int(6+16*z[0]), min_samples_leaf=int(2+18*z[1]), n_jobs=2, random_state=SEED)
    return HistGradientBoostingClassifier(max_iter=100, learning_rate=10**(-2+1.3*z[0]), max_leaf_nodes=int(7+25*z[1]), l2_regularization=1., random_state=SEED)

def dataset(samples):
    cache = ROOT/'data_cache'; cache.mkdir(exist_ok=True)
    path = cache/f'train-{samples}-features-v{FEATURE_SCHEMA_VERSION}.joblib'
    if path.exists():
        return joblib.load(path)
    xs, ys, groups = [], [], []
    for folder in sorted((ROOT/'train').iterdir()):
        if not folder.is_dir(): continue
        trip = read_trip(folder); net = Network(trip)
        ids, _, x = prepare(trip, net)
        reference = read_reference(folder, len(trip.arcs))
        target = proxy_labels(trip, net, reference)
        select = np.unique(np.linspace(0, len(x)-1, min(samples, len(x)), dtype=int))
        xs.append(x[select].reshape(-1, x.shape[-1]))
        ys.append((ids[select] == target[select, None]).ravel().astype(int))
        groups.extend([trip.id]*(len(select)*ids.shape[1]))
        print(f'Prepared train {trip.id}: {len(select)}/{len(x)} observations', flush=True)
    data = np.concatenate(xs), np.concatenate(ys), np.array(groups)
    joblib.dump(data, path)
    return data

def score_trip(model, folder, decoder_config, with_map=False):
    trip = read_trip(folder); net = Network(trip)
    start = time.perf_counter()
    names = LEGACY_FEATURE_NAMES if getattr(model, 'n_features_in_', 20) == 8 else FEATURE_NAMES
    ids, frac, x = prepare(trip, net, feature_names=names)
    p = probabilities(model, x)
    raw = ids[np.arange(len(ids)), p.argmax(axis=1)]
    decoder = Decoder(net, **decoder_config)
    predicted, resets = decoder.decode(trip.track, ids, frac, p)
    elapsed = time.perf_counter()-start
    # Only the evaluation layer opens the reference, after predictions finish.
    route = read_reference(folder, len(trip.arcs))
    target = proxy_labels(trip, net, route)
    y = (ids == target[:, None]).ravel().astype(int)
    binary = model.predict(x.reshape(-1, x.shape[-1]))
    raw_probability = model.predict_proba(x.reshape(-1, x.shape[-1]))
    precision, recall, f1, _ = precision_recall_fscore_support(y, binary, average='macro', zero_division=0)
    row = dict(trip=trip.id, observations=len(ids), top1_accuracy=float(np.mean(raw==target)), hmm_accuracy=float(np.mean(predicted==target)), candidate_miss_rate=float(np.mean(~(ids==target[:, None]).any(axis=1))), macro_precision=float(precision), macro_recall=float(recall), macro_f1=float(f1), candidate_accuracy=float(accuracy_score(y, binary)), log_loss=float(log_loss(y, raw_probability, labels=[0,1])), classifier_continuity=decoder.continuity(trip.track, ids, frac, raw), route_continuity=decoder.continuity(trip.track, ids, frac, predicted), sequence_exact_match=float(np.array_equal(predicted, target)), inference_seconds=elapsed, resets=len(resets), confusion_matrix=confusion_matrix(y, binary, labels=[0,1]).tolist())
    if with_map:
        from .maps import make_map
        make_map(trip, predicted, reference=route).save(str(OUT/'representative_map.html'))
    return row

def evaluate(model, config, with_map=False):
    rows = []
    for i, folder in enumerate(sorted((ROOT/'student_test').iterdir())):
        if folder.is_dir():
            print(f'Evaluating {folder.name}', flush=True)
            rows.append(score_trip(model, folder, config, with_map and i==0))
    keys = [k for k, v in rows[0].items() if isinstance(v, (int, float)) and k not in ('observations',)]
    mean = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    return mean, rows

def bayesian(name, x, y, groups, trials):
    rng = np.random.default_rng(SEED)
    zs, scores, history = [], [], []
    cv = GroupKFold(3)
    for i in range(trials):
        if i < 3:
            z = rng.random(2)
        else:
            gp = GaussianProcessRegressor(kernel=Matern(nu=2.5)+WhiteKernel(1e-5), normalize_y=True, random_state=SEED)
            gp.fit(np.array(zs), scores)
            pool = rng.random((256, 2)); mu, sd = gp.predict(pool, return_std=True)
            improvement = mu-max(scores)-0.001
            zscore = improvement/np.maximum(sd, 1e-12)
            ei = improvement*norm.cdf(zscore)+sd*norm.pdf(zscore)
            z = pool[np.argmax(ei)]
        model = tuned_model(name, z); start=time.perf_counter()
        score = float(cross_val_score(model, x, y, groups=groups, cv=cv, scoring='f1_macro', n_jobs=1).mean())
        zs.append(z); scores.append(score)
        history.append(dict(trial=i+1, unit_parameters=z.tolist(), cv_macro_f1=score, seconds=time.perf_counter()-start, configuration=str(model)))
        save_json(OUT/'bayesian_trials.json', history)
        print(f'Bayesian trial {i+1}/{trials}: {score:.4f}', flush=True)
    best=int(np.argmax(scores))
    return tuned_model(name, zs[best]), history[best]

def main():
    from .train import main as run_training
    return run_training()

if __name__ == '__main__': main()
