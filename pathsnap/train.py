"""Bounded, training-only grouped selection and one final student-test evaluation."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import argparse
import hashlib
import json
import time
import warnings
from pathlib import Path
import joblib
import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import precision_recall_fscore_support, log_loss, confusion_matrix
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from .data import ROOT, read_trip, read_reference, Network
from .features import prepare, FEATURE_NAMES, FEATURE_SCHEMA_VERSION
from .alignment import align_reference, ALIGNMENT_VERSION
from .decoder import Decoder

SEED=352
RESULTS=ROOT/'results/retrained'
CACHE=ROOT/'data_cache/retrained'
PROJECTION='WGS84 local azimuthal equidistant; always_xy'
SEARCH_SPACES={
    'LogisticRegression':dict(C=dict(scale='log10',minimum=.001,maximum=100.)),
    'GaussianNB':dict(var_smoothing=dict(scale='log10',minimum=1e-12,maximum=.001)),
    'DecisionTree':dict(max_depth=[4,24],min_samples_leaf=[2,32]),
    'RandomForest':dict(max_depth=[8,28],min_samples_leaf=[2,20],n_estimators=100),
    'HistGradientBoosting':dict(learning_rate=dict(scale='log',minimum=.02,maximum=.3),max_leaf_nodes=[15,63],l2_regularization=[0.,5.],max_iter=120,early_stopping=False),
}

def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def data_digest(records):
    h=hashlib.sha256()
    for r in records:
        h.update(r['trip'].encode());h.update(r['labels'].tobytes());h.update(r['x'].tobytes())
    return h.hexdigest()

def record(folder):
    """Cache includes raw-file fingerprints, feature and alignment versions."""
    files=[p for p in sorted(Path(folder).iterdir()) if p.suffix in ('.track','.nodes','.arcs','.route')]
    signature=str([(p.name,p.stat().st_size,p.stat().st_mtime_ns) for p in files])
    digest=hashlib.sha256((signature+f'features{FEATURE_SCHEMA_VERSION}-alignment{ALIGNMENT_VERSION}').encode()).hexdigest()[:16]
    path=CACHE/f'{Path(folder).parent.name}-{Path(folder).name}-{digest}.joblib'
    if path.exists():
        cached=joblib.load(path)
        if cached['x'].shape[-1]!=len(FEATURE_NAMES) or not np.isfinite(cached['x']).all():
            raise ValueError('Invalid cached feature schema or nonfinite values')
        return cached
    trip=read_trip(folder);net=Network(trip);start=time.perf_counter()
    ids,fractions,x=prepare(trip,net,k=12)
    if x.shape[-1]!=20 or not np.isfinite(x).all():raise ValueError(f'{trip.id}: invalid feature matrix')
    if (x[:,:,0]<0).any() or (fractions<0).any() or (fractions>1).any():raise ValueError('Invalid geometric features')
    feature_seconds=time.perf_counter()-start
    reference=read_reference(folder,len(trip.arcs));alignment=align_reference(trip,net,reference)
    out=dict(trip=trip.id,folder=str(folder),ids=ids.astype(np.int32),fractions=fractions.astype(np.float32),x=x.astype(np.float32),
             labels=alignment.labels.astype(np.int32),confident=alignment.confident,alignment_resets=alignment.resets,
             repeated_route_entries=int(len(reference)-len(np.unique(reference))),disconnected_reference_steps=alignment.disconnected_reference_steps,
             feature_seconds=feature_seconds,track=trip.track)
    joblib.dump(out,path,compress=1)
    return out

def families():
    return {
        'LogisticRegression':make_pipeline(StandardScaler(),LogisticRegression(C=1.,class_weight='balanced',max_iter=500,random_state=SEED)),
        'GaussianNB':make_pipeline(StandardScaler(),GaussianNB(var_smoothing=1e-8)),
        'DecisionTree':DecisionTreeClassifier(max_depth=14,min_samples_leaf=10,class_weight='balanced',random_state=SEED),
        'RandomForest':RandomForestClassifier(n_estimators=100,max_depth=18,min_samples_leaf=5,class_weight='balanced',n_jobs=2,random_state=SEED),
        'HistGradientBoosting':HistGradientBoostingClassifier(max_iter=120,max_leaf_nodes=31,learning_rate=.08,l2_regularization=1.,early_stopping=False,random_state=SEED),
    }

def matrix(records,indices,k,columns,samples):
    xs=[];ys=[]
    for i in indices:
        r=records[i];take=np.unique(np.linspace(0,len(r['x'])-1,min(samples,len(r['x'])),dtype=int))
        take=take[r['confident'][take]]
        if not len(take):continue
        x=r['x'][take,:k][:,:,columns]
        xs.append(x.reshape(-1,len(columns)))
        ys.append((r['ids'][take,:k]==r['labels'][take,None]).ravel().astype(np.int8))
    return np.concatenate(xs),np.concatenate(ys)

def positive(model,x):
    expected=getattr(model,'n_features_in_',None)
    if x.shape[-1]!=expected:raise ValueError(f'Model expects {expected} features, got {x.shape[-1]}')
    p=model.predict_proba(x.reshape(-1,x.shape[-1]))[:,list(model.classes_).index(1)]
    return np.clip(p.reshape(x.shape[:2]),1e-9,1-1e-9)

def metrics(r,raw,k):
    ids=r['ids'][:,:k];target=r['labels'];p=raw[:,:k]
    rank=np.argsort(-p,axis=1,kind='stable');pred=ids[np.arange(len(ids)),rank[:,0]]
    correct=pred==target;hit=(ids==target[:,None]);q=p/p.sum(1,keepdims=True)
    precision,recall,f1,_=precision_recall_fscore_support(target,pred,average='macro',zero_division=0)
    binary=hit.ravel().astype(int);bp=(p.ravel()>=.5).astype(int)
    mask=r['confident']
    return dict(top1_accuracy=float(correct.mean()),top2_accuracy=float(np.mean(np.any(np.take_along_axis(hit,rank[:,:min(2,k)],axis=1),axis=1))),
                top5_accuracy=float(np.mean(np.any(np.take_along_axis(hit,rank[:,:min(5,k)],axis=1),axis=1))),
                candidate_miss_rate=float(np.mean(~hit.any(1))),confident_coverage=float(mask.mean()),
                confident_top1_accuracy=float(correct[mask].mean()) if mask.any() else 0.,
                macro_precision=float(precision),macro_recall=float(recall),macro_f1=float(f1),
                log_loss=float(-np.log(np.maximum((q*hit).sum(1),1e-12)).mean()),
                candidate_log_loss=float(log_loss(binary,np.column_stack([1-p.ravel(),p.ravel()]),labels=[0,1])),
                candidate_confusion_matrix=confusion_matrix(binary,bp,labels=[0,1]).tolist())

def average(rows):
    return {k:float(np.mean([r[k] for r in rows])) for k,v in rows[0].items() if isinstance(v,(float,int))}

def grouped(model,records,columns,k,samples):
    predictions=[None]*len(records);fold_scores=[];start=time.perf_counter()
    for fold,(fit,valid) in enumerate(GroupKFold(3).split(np.arange(len(records)),groups=[r['trip'] for r in records])):
        x,y=matrix(records,fit,k,columns,samples);fitted=clone(model).fit(x,y);scores=[]
        for i in valid:
            predictions[i]=positive(fitted,records[i]['x'][:,:k][:,:,columns]).astype(np.float32)
            scores.append(metrics(records[i],predictions[i],k)['top1_accuracy'])
        fold_scores.append(float(np.mean(scores)))
    per_trip=[dict(trip=r['trip'],**metrics(r,p,k)) for r,p in zip(records,predictions)]
    return dict(cv_fold_accuracy=fold_scores,
                cv_std=float(np.std(fold_scores)),seconds=time.perf_counter()-start,**{'cv_'+key:value for key,value in average(per_trip).items()}),predictions,per_trip

def parameters(name,z):
    base=families()[name]
    if name=='LogisticRegression':base.set_params(logisticregression__C=float(10**(-3+5*z[0])))
    elif name=='GaussianNB':base.set_params(gaussiannb__var_smoothing=float(10**(-12+9*z[0])))
    elif name=='DecisionTree':base.set_params(max_depth=int(round(4+20*z[0])),min_samples_leaf=int(round(2+30*z[1])))
    elif name=='RandomForest':base.set_params(max_depth=int(round(8+20*z[0])),min_samples_leaf=int(round(2+18*z[1])))
    else:base.set_params(learning_rate=float(np.exp(np.log(.02)+z[0]*np.log(15))),max_leaf_nodes=int(round(15+48*z[1])),l2_regularization=float(5*z[2]))
    return base

def initial_point(name):
    return {'LogisticRegression':[.6],'GaussianNB':[4/9],'DecisionTree':[.5,8/30],
            'RandomForest':[.5,3/18],'HistGradientBoosting':[np.log(4)/np.log(15),16/48,.2]}[name]

def bayesian(name,records,columns,k,samples,trials):
    save(RESULTS/'search_space.json',dict(model=name,space=SEARCH_SPACES[name],objective='three-fold training-trip grouped all-observation Top-1 proxy accuracy',seed=SEED,trials=trials,acquisition='Gaussian-process Matern expected improvement; 512 proposals per acquisition'))
    rng=np.random.default_rng(SEED);zs=[];scores=[];history=[];best=None;best_predictions=None
    for i in range(trials):
        if i==0:z=np.array(initial_point(name))
        elif i<3:z=rng.random(len(zs[0]))
        else:
            gp=GaussianProcessRegressor(kernel=Matern(length_scale=np.ones(len(zs[0]))*.4,length_scale_bounds=(.05,4),nu=2.5),alpha=1e-5,normalize_y=True,random_state=SEED)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore');gp.fit(np.array(zs),scores)
            pool=rng.random((512,len(zs[0])));mu,sd=gp.predict(pool,return_std=True)
            improvement=mu-max(scores)-.0001;safe=np.maximum(sd,1e-12);zz=improvement/safe
            z=pool[np.argmax(improvement*norm.cdf(zz)+sd*norm.pdf(zz))]
        model=parameters(name,z);cv,predictions,_=grouped(model,records,columns,k,samples)
        score=cv['cv_top1_accuracy'];history.append(dict(trial=i,parameters=str(model),unit_parameters=z.tolist(),objective=score,seconds=cv['seconds']))
        if best is None or score>max(scores):best=model;best_predictions=predictions
        zs.append(z);scores.append(score)
        save(RESULTS/'bayesian_trials.json',history)
        print(f'Bayesian trial {i+1}/{trials}: train-grouped Top-1={score:.4f}',flush=True)
    return best,best_predictions,history

def decoder_selection(records,predictions,k):
    # Full sequences from nine deterministic out-of-fold training trips; never student tests.
    eligible=sorted([i for i,r in enumerate(records) if len(r['track'])<=5000],key=lambda i:len(records[i]['track']))
    selected=[eligible[i] for i in np.unique(np.linspace(0,len(eligible)-1,min(9,len(eligible)),dtype=int))]
    settings=[]
    for beta in (30.,90.):
        for speed in (40.,70.):
            for gap in (20.,60.):
                for weight in (.1,.5):settings.append(dict(beta=beta,max_speed=speed,gap_reset=gap,emission_weight=1.,transition_weight=weight))
    rows=[dict(parameters=scores,accuracy=[],continuity=[]) for scores in settings]
    chosen_details=[]
    for i in selected:
        r=records[i];trip=read_trip(r['folder']);net=Network(trip);ids=r['ids'][:,:k];frac=r['fractions'][:,:k]
        base=Decoder(net);lattice=base.lattice(trip.track,ids,frac);p=predictions[i];p=p/p.sum(1,keepdims=True)
        for row in rows:
            decoder=Decoder(net,**row['parameters']);result,resets=decoder.decode_lattice(ids,p,lattice)
            row['accuracy'].append(float(np.mean(result==r['labels'])))
            row['continuity'].append(lattice_continuity(ids,result,lattice,decoder))
        chosen_details.append(dict(trip=r['trip'],observations=len(ids),alignment_confident_coverage=float(r['confident'].mean())))
    for row in rows:
        row['mean_top1_accuracy']=float(np.mean(row.pop('accuracy')))
        row['mean_continuity']=float(np.mean(row.pop('continuity')))
    winner=max(rows,key=lambda r:(r['mean_top1_accuracy'],r['mean_continuity']))
    save(RESULTS/'decoder_validation.json',dict(partition='training_out_of_fold',trips=chosen_details,trials=rows,chosen=winner))
    return winner['parameters']

def lattice_continuity(ids,chosen,lattice,decoder):
    states=np.argmax(ids==chosen[:,None],axis=1);valid=[]
    for t,(roads,dt) in enumerate(zip(lattice['roads'],lattice['dt']),1):
        if dt>decoder.gap_reset or roads is None:continue
        valid.append(bool(roads[states[t-1],states[t]]<=min(10000,max(80,decoder.max_speed*dt+60))))
    return float(np.mean(valid)) if valid else 1.

def final_evaluation(model,records,columns,k,decoder_config):
    rows=[]
    from .maps import make_map
    for r in records:
        start=time.perf_counter();p=positive(model,r['x'][:,:k][:,:,columns]);rank=p.argmax(1)
        raw=r['ids'][np.arange(len(p)),rank];trip=read_trip(r['folder']);net=Network(trip);decoder=Decoder(net,**decoder_config)
        ids=r['ids'][:,:k];lattice=decoder.lattice(trip.track,ids,r['fractions'][:,:k])
        predicted,resets=decoder.decode_lattice(ids,p/p.sum(1,keepdims=True),lattice)
        m=metrics(r,p,k);mask=r['confident']
        m.update(trip=r['trip'],observations=len(p),hmm_accuracy=float(np.mean(predicted==r['labels'])),
                 hmm_confident_accuracy=float(np.mean(predicted[mask]==r['labels'][mask])) if mask.any() else 0.,
                 classifier_continuity=lattice_continuity(ids,raw,lattice,decoder),hmm_continuity=lattice_continuity(ids,predicted,lattice,decoder),
                 sequence_exact_match=float(np.array_equal(predicted,r['labels'])),resets=len(resets),inference_seconds=time.perf_counter()-start,
                 preprocessing_seconds=r['feature_seconds'],alignment_resets=len(r['alignment_resets']))
        rows.append(m)
        if r['trip']==records[0]['trip']:make_map(trip,predicted).save(str(RESULTS/'predicted_route.html'))
        print(f'Student trip {r["trip"]}: classifier={m["top1_accuracy"]:.4f}, HMM={m["hmm_accuracy"]:.4f}, confident coverage={m["confident_coverage"]:.3f}',flush=True)
    total=sum(r['observations'] for r in rows)
    pooled={key:float(sum(r[key]*r['observations'] for r in rows)/total) for key in ('top1_accuracy','hmm_accuracy')}
    return dict(label_method='ordered-reference alignment proxy, not independently verified GPS observation ground truth',aggregation='mean: equal weight per trip; pooled_observations: equal weight per GPS observation',mean=average(rows),pooled_observations=pooled,total_observations=total,trips=rows)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--samples-per-trip',type=int,default=256);parser.add_argument('--trials',type=int,default=6)
    args=parser.parse_args()
    if args.samples_per_trip<32 or args.trials<4:parser.error('Use at least 32 samples and 4 Bayesian trials')
    RESULTS.mkdir(parents=True,exist_ok=True);CACHE.mkdir(parents=True,exist_ok=True)
    folders=sorted(p for p in (ROOT/'train').iterdir() if p.is_dir())
    student_folders=sorted(p for p in (ROOT/'student_test').iterdir() if p.is_dir())
    if len(folders)!=80 or len(student_folders)!=10:raise ValueError('Expected 80 training and 10 student-test trips')
    if {p.name for p in folders}&{p.name for p in student_folders}:raise ValueError('Trip leakage')
    records=[]
    for i,folder in enumerate(folders):
        records.append(record(folder))
        if (i+1)%10==0:print(f'Prepared {i+1}/80 training trips',flush=True)
    alignment_summary=[dict(trip=r['trip'],observations=len(r['x']),confident_coverage=float(r['confident'].mean()),alignment_resets=len(r['alignment_resets']),repeated_route_entries=r['repeated_route_entries'],disconnected_reference_steps=r['disconnected_reference_steps']) for r in records]
    save(RESULTS/'alignment_training.json',alignment_summary)
    base=families()['HistGradientBoosting'];ablations=[];ab_predictions={}
    for count in (8,12,20):
        cv,p,_=grouped(base,records,list(range(count)),12,args.samples_per_trip)
        ablations.append(dict(features=count,**cv));ab_predictions[count]=p
        save(RESULTS/'feature_ablation.json',ablations);print(f'Feature ablation {count}: grouped Top-1={cv["cv_top1_accuracy"]:.4f}',flush=True)
    scores={r['features']:r['cv_top1_accuracy'] for r in ablations}
    count=20 if scores[20]>=scores[12]-.002 else 12;columns=list(range(count))
    candidate_rows=[]
    for k in (4,6,8,10,12):
        start=time.perf_counter();ms=[metrics(r,p,k) for r,p in zip(records,ab_predictions[count])]
        candidate_rows.append(dict(k=k,ranking_seconds=time.perf_counter()-start,**average(ms)))
    k=max(candidate_rows,key=lambda r:(r['top1_accuracy'],-r['k']))['k']
    # Fresh spatial extraction timing by k on two deterministic TRAINING trips.
    for row in candidate_rows:
        elapsed=[]
        for i in (0,len(records)//2):
            t=read_trip(records[i]['folder']);n=Network(t);start=time.perf_counter();n.candidates(t.track,k=row['k']);elapsed.append(time.perf_counter()-start)
        row['fresh_candidate_seconds_two_training_trips']=elapsed
    save(RESULTS/'candidate_comparison.json',candidate_rows)
    print(f'Training-only schema selection: {count} features, k={k}',flush=True)
    comparison=[];all_predictions={}
    prior_folder=RESULTS.parent/'retrained_initial'
    prior_config=json.loads((prior_folder/'selection.json').read_text()) if (prior_folder/'selection.json').exists() else {}
    prior_rows={r['model']:r for r in json.loads((prior_folder/'model_comparison.json').read_text())} if prior_config else {}
    reusable=(prior_config.get('dataset_digest')==data_digest(records) and prior_config.get('k')==k
              and prior_config.get('feature_names')==[FEATURE_NAMES[j] for j in columns]
              and prior_config.get('samples_per_trip')==args.samples_per_trip and prior_config.get('seed')==SEED)
    for name,model in families().items():
        if reusable and prior_rows.get(name,{}).get('configuration')==str(model):
            row=dict(prior_rows[name],cv_reused_exact_matching_run=True)
        else:
            cv,p,per_trip=grouped(model,records,columns,k,args.samples_per_trip)
            row=dict(model=name,configuration=str(model),cv_reused_exact_matching_run=False,**cv);all_predictions[name]=p
            save(RESULTS/f'{name}_cv_trips.json',per_trip)
        comparison.append(row)
        save(RESULTS/'model_comparison.json',comparison)
        print(f'{name}: training-grouped Top-1={row["cv_top1_accuracy"]:.4f}',flush=True)
    name=max(comparison,key=lambda r:r['cv_top1_accuracy'])['model']
    model,predictions,trials=bayesian(name,records,columns,k,args.samples_per_trip,args.trials)
    joblib.dump(dict(predictions=predictions,k=k,digest=data_digest(records),feature_schema_version=FEATURE_SCHEMA_VERSION),CACHE/'selected_oof.joblib',compress=1)
    decoder_config=decoder_selection(records,predictions,k)
    x,y=matrix(records,np.arange(len(records)),k,columns,args.samples_per_trip)
    validation=dict(feature_names=[FEATURE_NAMES[j] for j in columns],finite=bool(np.isfinite(x).all()),
                    minimum=x.min(0).tolist(),maximum=x.max(0).tolist(),standard_deviation=x.std(0).tolist(),
                    nonconstant_features=int(np.sum(x.std(0)>1e-8)))
    if validation['nonconstant_features']<12:raise ValueError('Fewer than twelve nonconstant model inputs')
    save(RESULTS/'feature_validation.json',validation)
    start=time.perf_counter();model.fit(x,y);refit_seconds=time.perf_counter()-start
    schema=[FEATURE_NAMES[j] for j in columns]
    schema_hash=hashlib.sha256(json.dumps(dict(names=schema,version=FEATURE_SCHEMA_VERSION,projection=PROJECTION),sort_keys=True).encode()).hexdigest()
    model.pathsnap_schema_hash_=schema_hash
    config=dict(model=name,configuration=str(model),feature_names=schema,feature_schema_version=FEATURE_SCHEMA_VERSION,feature_schema_hash=schema_hash,projection=PROJECTION,
                alignment_version=ALIGNMENT_VERSION,label_method='ordered-reference alignment proxy with uncertainty flags; not verified observation ground truth',
                k=k,decoder=decoder_config,seed=SEED,training_trips=80,student_test_trips=10,samples_per_trip=args.samples_per_trip,
                training_candidate_rows=len(y),selection_metric='all-observation Top-1 aligned-proxy accuracy; three-fold training-trip GroupKFold',
                cv_top1_accuracy=max(t['objective'] for t in trials),refit_seconds=refit_seconds,parameters={key:value for key,value in model.get_params().items() if isinstance(value,(int,float,str,bool,type(None)))})
    save(RESULTS/'selection.json',config)
    # Only NOW open student-test observations/references, after all choices freeze.
    students=[record(folder) for folder in student_folders]
    final=final_evaluation(model,students,columns,k,decoder_config)
    final['target95_on_aligned_proxy']=final['mean']['top1_accuracy']>=.95
    final['verified_ground_truth_available']=False
    save(RESULTS/'student_test.json',final)
    # Measure all five refitted families once, with no further selection on student data.
    student_comparison=[]
    old_student={r['model']:r for r in json.loads((prior_folder/'student_model_comparison.json').read_text())} if prior_config else {}
    same_students=prior_config.get('student_digest')==data_digest(students)
    for row in comparison:
        if same_students and row['cv_reused_exact_matching_run'] and row['model'] in old_student:
            student_comparison.append(dict(old_student[row['model']],reused_exact_matching_run=True))
        else:
            baseline=families()[row['model']].fit(x,y)
            ms=[metrics(r,positive(baseline,r['x'][:,:k][:,:,columns]),k) for r in students]
            student_comparison.append(dict(model=row['model'],reused_exact_matching_run=False,**average(ms)))
    save(RESULTS/'student_model_comparison.json',student_comparison)
    art=ROOT/'artifacts';art.mkdir(exist_ok=True)
    joblib.dump(model,art/'model.joblib');save(art/'config.json',config)
    save(art/'feature_schema.json',dict(version=FEATURE_SCHEMA_VERSION,names=schema,hash=schema_hash,projection=PROJECTION))
    print(f'Final saved: {name}, {len(schema)} features, classifier={final["mean"]["top1_accuracy"]:.4f}, HMM={final["mean"]["hmm_accuracy"]:.4f}',flush=True)

def recalibrate():
    """Reproduce decoder tuning from saved TRAINING OOF predictions only."""
    oof=joblib.load(CACHE/'selected_oof.joblib')
    records=[record(p) for p in sorted((ROOT/'train').iterdir()) if p.is_dir()]
    if oof['feature_schema_version']!=FEATURE_SCHEMA_VERSION or oof['digest']!=data_digest(records):
        raise ValueError('Training data/schema changed; rerun grouped training')
    cfg=json.loads((ROOT/'artifacts/config.json').read_text())
    selected=json.loads((RESULTS/'selection.json').read_text())
    if cfg['configuration']!=selected['configuration'] or cfg['feature_schema_hash']!=selected['feature_schema_hash'] or cfg['k']!=oof['k']:
        raise ValueError('Saved training OOF predictions do not match the active model')
    cfg['decoder']=decoder_selection(records,oof['predictions'],oof['k'])
    save(ROOT/'artifacts/config.json',cfg)

if __name__=='__main__':main()
