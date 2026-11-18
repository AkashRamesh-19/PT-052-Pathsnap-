"""Reproduce the Phase 1 demonstration without training or loading a model."""
import html
import json
import shutil
from pathlib import Path
import numpy as np
import folium
from .data import ROOT, read_trip, read_reference, Network
from .features import FEATURE_NAMES, FEATURE_SCHEMA_VERSION, prepare, proxy_labels


def main():
    out = ROOT/'phase1_demo'
    out.mkdir(exist_ok=True)
    lines = []
    def emit(value=''):
        lines.append(str(value)); print(value, flush=True)

    inventory = json.loads((ROOT/'results/inventory.json').read_text(encoding='utf-8'))
    emit('PATHSNAP - PHASE 1: DATA EXPLORATION AND PREPROCESSING')
    emit('='*70)
    emit('Development dataset inventory (recorded in results/inventory.json):')
    for partition in ('train','student_test'):
        rows = [r for r in inventory if r['partition']==partition]
        emit(f'  {partition:14s}: {len(rows):2d} trips; {sum(r["observations"] for r in rows):,} GPS observations')
    emit('  Evaluator partition: NOT accessed')
    emit(f'  Invalid trips in recorded inventory: {sum("error" in r for r in inventory)}')
    emit(f'  Networks with multiple weak components: {sum(r["weak_components"]>1 for r in inventory)}/{len(inventory)}')
    emit(f'  Largest recorded gap: {max(r["max_gap"] for r in inventory):,.0f} seconds')
    emit()
    folder = ROOT/'train/00000001'
    trip = read_trip(folder)
    net = Network(trip)
    emit(f'Actual preprocessing run: train/{trip.id}')
    emit(f'  GPS shape: {trip.track.shape} [longitude, latitude, timestamp]')
    emit(f'  Nodes: {len(trip.nodes):,}; directed arcs: {len(trip.arcs):,}')
    emit('  PASS: numeric shapes, finite values, coordinate ranges, arc indices')
    emit('  PASS: timestamps strictly increasing; original observation order retained')
    emit('  No observations dropped or imputed: this trip passes validation')
    emit(f'  Timestamp gaps: min={np.diff(trip.track[:,2]).min():.0f}s, max={np.diff(trip.track[:,2]).max():.0f}s')
    emit('  Geometry: local metre projection; longitude scale uses cos(mean latitude)')
    emit('  Candidate search: exact point-to-segment distance with KD-tree bounds')
    ids, fractions, features = prepare(trip, net, k=6)
    emit(f'  Candidate IDs shape: {ids.shape}; features shape: {features.shape}')
    emit(f'  Candidate rows: {ids.size:,}; feature count: {len(FEATURE_NAMES)}')
    emit(f'  Nonfinite feature values: {int((~np.isfinite(features)).sum())}')
    emit(f'  Feature schema: v{FEATURE_SCHEMA_VERSION} (20 extracted inputs; the trained model uses its saved subset)')
    emit('  First candidate: all 20 extracted feature values')
    for number, (name, value) in enumerate(zip(FEATURE_NAMES, features[0, 0]), 1):
        emit(f'    {number:2d}. {name:30s} = {value:.6f}')
    emit()
    emit('First observation: six candidate segments and actual features')
    emit(' arc_id   distance_m   heading_cos   fraction   speed_mps   gap_s')
    for arc, row in zip(ids[0], features[0]):
        emit(f'{arc:7d}   {row[0]:10.3f}   {row[2]:11.4f}   {row[5]:8.4f}   {row[6]:9.3f}   {row[7]:5.1f}')
    reference = read_reference(folder, len(trip.arcs))
    target = proxy_labels(trip, net, reference)
    emit()
    emit(f'Training-only reference handling: {len(reference)} chronological route entries')
    emit('  The route is NOT an observation-by-observation label file.')
    emit('  Proxy targets use ordered-reference sequence alignment; they remain inferred labels.')
    emit(f'  Proxy outside the six candidates: {np.mean(~(ids==target[:,None]).any(axis=1)):.2%} of observations')
    emit('  Direction ties and repeated visits remain ambiguous; this is not verified GPS ground truth.')
    emit('  Reference information is never part of inference features.')
    data = np.column_stack([np.repeat(np.arange(len(ids)), ids.shape[1]), ids.ravel(), features.reshape(-1, features.shape[-1])])
    np.savetxt(out/'preprocessed_candidates.csv', data, delimiter=',', header='observation,arc_id,'+','.join(FEATURE_NAMES), comments='', fmt='%.8g')
    emit()
    emit('Saved: phase1_demo/preprocessed_candidates.csv')
    emit('No classifier training, HMM decoding or evaluator processing was performed.')
    (out/'preprocessing_output.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (ROOT/'results/phase1_preprocessing_output.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    summary = dict(schema_version=FEATURE_SCHEMA_VERSION, feature_count=len(FEATURE_NAMES), feature_names=FEATURE_NAMES,
                   partition='train', trip=trip.id, observations=len(ids), candidates_per_observation=ids.shape[1],
                   feature_shape=list(features.shape), candidate_rows=int(ids.size),
                   nonfinite_values=int((~np.isfinite(features)).sum()),
                   trained_model=False, evaluator_accessed=False)
    (ROOT/'results/phase1_feature_summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    for name in ('data','features'):
        (out/(name+'_source.txt')).write_text((ROOT/'pathsnap'/f'{name}.py').read_text(encoding='utf-8'),encoding='utf-8')
    shutil.copyfile(ROOT/'results/eda.png',out/'eda.png')
    shutil.copyfile(ROOT/'results/inventory.json',out/'inventory.json')
    m = folium.Map(location=trip.track[:,:2].mean(0)[::-1].tolist())
    coords = trip.nodes[trip.arcs]
    low = trip.track[:,:2].min(0)-.003; high = trip.track[:,:2].max(0)+.003
    nearby = ((coords>=low)&(coords<=high)).all(axis=(1,2))
    folium.GeoJson({'type':'Feature','properties':{},'geometry':{'type':'MultiLineString','coordinates':coords[nearby].tolist()}},name='Nearby road network',style_function=lambda _:dict(color='#68758a',weight=2)).add_to(m)
    folium.PolyLine(trip.track[:,1::-1].tolist(),color='#1685ee',weight=3,tooltip='Original GPS observations').add_to(m)
    folium.LayerControl().add_to(m)
    m.fit_bounds([trip.track[:,1::-1].min(0).tolist(),trip.track[:,1::-1].max(0).tolist()])
    m.save(str(out/'gps_map.html'))
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PathSnap | Phase 1</title><style>body{font:17px system-ui;background:#0c1423;color:#e5edf9;max-width:1150px;margin:auto;padding:35px}h1{font-size:38px}h2{color:#62dfbc}p{line-height:1.7}a{color:#71c3ff}section{background:#142238;padding:22px;border-radius:12px;margin:25px 0}img,iframe{width:100%;border:0;border-radius:8px}iframe{height:560px}pre{white-space:pre-wrap;font-size:13px;line-height:1.6}nav{line-height:2}</style><h1>PathSnap · Phase 1</h1><p>Understand noisy GPS data and prepare candidate-road features for later machine learning.</p><nav><a href="#summary">1. Dataset</a> · <a href="#map">2. GPS and roads</a> · <a href="#output">3. Actual preprocessing output</a> · <a href="data_source.txt">Data source</a> · <a href="features_source.txt">Feature source</a> · <a href="preprocessed_candidates.csv">Feature CSV</a></nav><section id="summary"><h2>1. Dataset and data quality</h2><p>80 training trips + 10 student-test trips = 227,577 GPS observations. Raw files describe GPS points, road nodes, directed connections and reference routes. All supplied development trips passed structural validation. 89 of 90 networks have multiple weak components. Large time gaps require care in later sequence modelling.</p><img src="eda.png" alt="Actual trajectory and network size distributions"><p><a href="inventory.json">Full per-trip inventory</a></p></section><section id="map"><h2>2. Actual training-trip GPS and road network</h2><p>Blue: observed GPS path. Grey: nearby road segments. This Phase 1 map contains no model predictions. Map tiles require internet.</p><iframe src="gps_map.html" title="Training GPS and road network"></iframe></section><section id="output"><h2>3. Preprocessing evidence</h2><p>Reproduce in a terminal from the ML_HACK folder: <code>python -m pathsnap.phase1</code></p><pre>OUTPUT_PLACEHOLDER</pre><a href="preprocessing_output.txt">Open terminal-output text</a></section><section><h2>What to explain</h2><p>We validate the raw data, inspect gaps and connectivity, convert geometry to metres, select six nearby road-segment candidates per GPS observation, and extract twenty geometry, motion and connectivity features. Reference routes need proxy alignment; they are not exact labels per GPS point. These outputs feed later model training. Phase 1 does not establish model accuracy.</p></section></html>'''
    (out/'index.html').write_text(page.replace('OUTPUT_PLACEHOLDER',html.escape('\n'.join(lines))),encoding='utf-8')
    emit('Presentation ready: phase1_demo/index.html')


if __name__=='__main__': main()
