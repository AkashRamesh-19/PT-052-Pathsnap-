import json
import logging
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import time
import hashlib
from functools import lru_cache
from pathlib import Path
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from .data import ROOT, read_trip, read_reference, Network
from .features import prepare, probabilities, FEATURE_NAMES, FEATURE_SCHEMA_VERSION
from .decoder import Decoder
from .maps import make_map
from .reconstruct import reconstruct_sections

log=logging.getLogger('uvicorn.error')
app=FastAPI(title='PathSnap',version='1.0.0')
# Local two-laptop demo; browser clients send no credentials.
app.add_middleware(CORSMiddleware,allow_origins=os.getenv('PATHSNAP_UI_ORIGINS','*').split(','),allow_methods=['GET','POST'],allow_headers=['Content-Type'])

def partition():
    name=os.getenv('PATHSNAP_PARTITION','student_test')
    if name not in ('student_test','evaluator_test'): raise HTTPException(500,'Invalid server partition configuration')
    return name

@lru_cache(maxsize=1)
def assets():
    path=ROOT/'artifacts/model.joblib'
    if not path.exists(): raise HTTPException(503,'Run python -m pathsnap.experiment first')
    # Load only the locally generated trusted artifact; never accept uploads.
    try:
        model=joblib.load(path);config=json.loads((ROOT/'artifacts/config.json').read_text())
    except (OSError,ValueError,EOFError) as exc:
        raise HTTPException(503,'Model/configuration cannot be loaded; retrain or restore matching artifacts') from exc
    if not {'feature_names','k','decoder','feature_schema_hash','feature_schema_version'}<=config.keys():
        raise HTTPException(503,'Incomplete trained-model configuration')
    if config.get('feature_schema_version')!=FEATURE_SCHEMA_VERSION:
        raise HTTPException(503,'Feature implementation changed: retrain using python -m pathsnap.train')
    if len(config['feature_names'])!=model.n_features_in_ or len(config['feature_names'])<12:
        raise HTTPException(503,'Model/feature schema mismatch')
    if any(name not in FEATURE_NAMES for name in config['feature_names']):
        raise HTTPException(503,'Unknown model feature schema')
    actual_hash=hashlib.sha256(json.dumps(dict(names=config['feature_names'],version=FEATURE_SCHEMA_VERSION,projection='WGS84 local azimuthal equidistant; always_xy'),sort_keys=True).encode()).hexdigest()
    if actual_hash!=config.get('feature_schema_hash') or getattr(model,'pathsnap_schema_hash_',None)!=actual_hash:
        raise HTTPException(503,'Model and preprocessing schema hashes disagree')
    return model,config

def folder_for(trip_id):
    if len(trip_id)!=8 or not trip_id.isascii() or not trip_id.isdigit(): raise HTTPException(404,'Unknown trip')
    folder=ROOT/partition()/trip_id
    if not folder.is_dir(): raise HTTPException(404,'Unknown trip')
    return folder

def infer(folder):
    model, config=assets()
    trip=read_trip(folder); network=Network(trip)
    ids,frac,x=prepare(trip,network,config['k'],feature_names=config['feature_names'])
    # Match the cached training tensors exactly, including numeric precision.
    x=x.astype(np.float32);frac=frac.astype(np.float32)
    p=probabilities(model,x)
    decoder=Decoder(network,**config['decoder'])
    result,resets=decoder.decode(trip.track,ids,frac,p)
    return trip,network,result,resets

@app.get('/health')
def health():
    try:
        model,config=assets();ready=True;detail=None
    except HTTPException as exc:
        ready=False;config={};detail=exc.detail
    return dict(status='ok',partition=partition(),model_ready=ready,feature_count=len(config.get('feature_names',[])),model=config.get('model'),detail=detail)

@app.get('/trips')
def trips():
    root=ROOT/partition()
    options=sorted(p.name for p in root.iterdir() if p.is_dir() and len(p.name)==8 and p.name.isascii() and p.name.isdigit()) if root.exists() else []
    return dict(partition=partition(),trips=options,expected_count=10,ready=len(options)==10)

@app.post('/predict/{trip_id}')
def predict(trip_id: str):
    folder=folder_for(trip_id); start=time.perf_counter()
    try:
        trip,network,predicted,resets=infer(folder)
        decoder_config=assets()[1]['decoder']
        sections=reconstruct_sections(trip,network,predicted,resets,max_speed=decoder_config['max_speed'],gap_reset=decoder_config['gap_reset'])
        metrics={}
        if partition()=='student_test' and os.getenv('PATHSNAP_SCORE_STUDENT','0')=='1' and list(folder.glob('*.route')):
            # Reference access occurs only AFTER inference and returns aggregates.
            reference=read_reference(folder,len(trip.arcs))
            from .alignment import align_reference
            alignment=align_reference(trip,network,reference)
            metrics=dict(proxy_pointwise_accuracy=float(np.mean(predicted==alignment.labels)),alignment_confident_coverage=float(alignment.confident.mean()),label_method='ordered-reference alignment proxy, not verified ground truth')
        html=make_map(trip,predicted,route_sections=sections).get_root().render()
        elapsed=time.perf_counter()-start
        log.info('prediction partition=%s trip=%s observations=%s seconds=%.3f resets=%s metrics=%s',partition(),trip_id,len(predicted),elapsed,len(resets),metrics)
        return dict(trip_id=trip_id,partition=partition(),observations=len(predicted),predicted_arc_ids=predicted.tolist(),route_sections=sections,resets=resets,seconds=elapsed,metrics=metrics,map_html=html)
    except (ValueError,OSError) as exc:
        log.exception('Trip processing failed: %s',trip_id)
        raise HTTPException(422,str(exc)) from exc

@app.get('/',response_class=HTMLResponse)
def interface():
    return (ROOT/'ui/index.html').read_text(encoding='utf-8')
