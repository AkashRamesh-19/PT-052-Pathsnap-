"""Package only current verified source/artifacts into disjoint team contributions."""
import hashlib
from importlib.metadata import requires,version
import json
from pathlib import Path
import re
import subprocess
import tempfile
import zipfile
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT=Path(__file__).resolve().parent
VERIFY=ROOT/'results/final_verification'
DEST=ROOT/'team_packages/CURRENT_VERIFIED_2026-11-18'

def decode(path):
    b=Path(path).read_bytes()
    return b.decode('utf-16' if b.startswith((b'\xff\xfe',b'\xfe\xff')) else 'utf-8').replace('\r\n','\n')

def lock_dependencies():
    queue=[Requirement(line).name for line in (ROOT/'requirements.txt').read_text().splitlines() if line.strip() and not line.startswith('#')];found={}
    while queue:
        name=queue.pop();key=canonicalize_name(name)
        if key in found:continue
        found[key]=version(name)
        for text in requires(name) or []:
            dep=Requirement(text)
            if dep.marker is None or dep.marker.evaluate({'extra':''}):queue.append(dep.name)
    (ROOT/'requirements-lock.txt').write_text('# Verified Windows / Python 3.12 runtime; preserve versions for the saved model.\n'+'\n'.join(k+'=='+v for k,v in sorted(found.items()))+'\n',encoding='utf-8')

def report():
    fresh=json.loads((VERIFY/'fresh_metrics.json').read_text());browser=json.loads((VERIFY/'browser_e2e.json').read_text());http=json.loads((VERIFY/'http.json').read_text());policy=json.loads((VERIFY/'training_policy.json').read_text());cfg=json.loads((ROOT/'artifacts/config.json').read_text())
    assert fresh['status']==http['status']=='PASS' and browser['real_browser'] and len(browser['trips'])>=2
    assert all(t.get('actual_bounds_contain_gps') and t.get('loaded_basemap_tiles',0)>0 for t in browser['trips'])
    unit=(VERIFY/'unit_tests.txt').read_text();count=int(re.search(r'Ran (\d+) tests',unit).group(1));assert '\nOK\n' in unit
    metrics=fresh['mean_per_trip'];p=fresh['pooled'];bayes=policy['bayesian']
    text=f'''# Final PathSnap execution verification

Verdict: **READY WITH MANUAL CHECKS**. Software executed on one Windows machine; this is not proof of physical two-laptop or reserved-evaluator execution.

| Component | Actually executed? | Evidence | Expected result | Actual result | Remaining issue |
|---|---|---|---|---|---|
| ML artifacts / features | PASS | training_policy.json; artifacts/config.json; unit_tests.txt | Matching schema and >=12 used inputs | {len(cfg['feature_names'])} inputs and split features; model reload predictions identical | Proxy labels are inferred, not verified truth |
| Five-model experiments / Bayesian selection | PASS | results/retrained/model_comparison.json; bayesian_trials.json; training_policy.json | >=5 models; genuine training-only optimisation | Five models; {bayes['trials']} trials; {bayes['gp_expected_improvement_proposals']} GP/EI proposals replay exactly; best parameters match final model | Student data has historical development use; not a pristine holdout |
| Data partitions / grouped CV | PASS | training_policy.json | 80 train / 10 student, disjoint trip folds | Fresh parsing of training data; disjoint 3-fold trip groups | Reserved data absent |
| HMM + Viterbi | PASS | fresh_metrics.json; predictions/*.npz | Real classifier probabilities, directed paths, gap handling | All 19,160 observations recalculated; 16 decoder settings on nine training OOF trips | HMM does not improve every trip |
| FastAPI backend | PASS | http.json; http_requests.txt; backend_log_excerpt.txt | Live model and real JSON predictions | 10/10 student requests match independent predictions; 404/405 errors and CORS checked | Student trips are NOT the reserved trips |
| Frontend + Folium | PASS | browser_e2e.json; ui_*.png; map_*.png | Real browser POSTs, GPS/roads/prediction, legend, bounds | Two trips, real loaded tiles, GPS bounds, mobile layout and invalid-URL recovery | Internet needed for external map resources |
| End-to-end pipeline | PASS | browser_e2e.json; http.json; fresh_metrics.json | UI selection through dynamic ML/HMM to map | No mocked API; model output matches fresh reference-free inference | None found in tested development flow |
| Two-laptop integration | NOT VERIFIED | Both listeners bind 0.0.0.0:8000/8080; CORS HTTP preflight passes | Successful requests from a second physical laptop | Separate API/frontend processes verified on one machine | Test actual LAN/firewall/device connection |
| Reserved evaluator trips | NOT VERIFIED | No evaluator_test folder; inference-leakage unit tests pass | Run actual ten reserved trips without opening hidden references | No reserved data or labels accessed | Obtain and run only under final-evaluation rules |
| Current required plots | PASS | results/retrained/cv_macro_f1.png; student_accuracy_f1.png; student_log_loss.png | Actual five-model comparison plots | Generated from current recorded JSON metrics | Include these in the team's report |
| GitHub / consolidated report | FAIL | remote_files.txt; historical PDF noted in README | Current full source, artifacts and current report | Remote main contained Part 1 only when fetched; old PDF is not a current report | Team must push assigned ZIP payloads, add details and update report; nothing published here |

## Confirmed results

- Final model: Bayesian-tuned HistGradientBoosting; {len(cfg['feature_names'])} actual inputs: {', '.join(cfg['feature_names'])}.
- Classifier: **{metrics['top1_accuracy']:.2%} mean per trip / {p['classifier']:.2%} pooled**.
- HMM: **{metrics['hmm_accuracy']:.2%} mean per trip / {p['hmm']:.2%} pooled**. Only the trip-mean HMM result exceeds 90%.
- Denominators: 10 equally weighted trips for means; {fresh['total_observations']:,} GPS observations for pooled scores. Candidate misses stay in denominators. Candidate miss rate: {metrics['candidate_miss_rate']:.2%}; HMM continuity: {metrics['hmm_continuity']:.2%}; sequence exact-match: {metrics['sequence_exact_match']:.2%}. Continuity excludes configured long gaps.
- Observation Macro F1: {metrics['macro_f1']:.6f}; observation log loss: {metrics['log_loss']:.6f}. Candidate-level precision/recall/F1, accuracy and confusion matrices are retained per trip in fresh_metrics.json.
- Bayesian parameters: learning_rate={cfg['parameters']['learning_rate']}, max_leaf_nodes={cfg['parameters']['max_leaf_nodes']}, L2={cfg['parameters']['l2_regularization']}, iterations={cfg['parameters']['max_iter']}, early_stopping=False, seed=352.
- Automated tests: **{count} passed, 0 failed**. ML weights unchanged during this audit. No training, student-driven parameter changes, evaluator-label access or GitHub publication performed.
- Historical development used student data in earlier experiments. Current final selection/tuning is training-only; do not describe student scores as a pristine independent holdout.

## Actual command and response evidence

```text
python -m pathsnap.audit
python -m pathsnap.audit_http
python -m unittest discover -s tests -v
python tests/browser_e2e.py --trips 2 --screenshots --output results/final_verification
GET /health -> 200; model_ready=true, feature_count=12
POST /predict/00000000 -> 200; observations=2503
POST /predict/not-a-trip -> 404
POST /predict/99999999 -> 404
GET /predict/00000000 -> 405
Cross-origin POST preflight -> 200
```

Real receiving-server log excerpt: backend_log_excerpt.txt. Full HTTP summaries: http_requests.txt. First predicted arc IDs, durations and per-trip response summaries: http.json. Independently saved states, probabilities and aligned labels: predictions/*.npz. No static route output was substituted.

Screenshots (real browser, individual trip scores are not global accuracy):
'''
    text+='\n'.join('- '+name for name in browser['screenshots'])
    text+='''

## Commands for the professor's demo

From the extracted full project root (or the integrated team repository), install matching dependencies once:

```powershell
python -m pip install -r requirements-lock.txt
python -m unittest discover -s tests -v
```

Laptop 1 / API terminal:

```powershell
$env:PATHSNAP_PARTITION='student_test'
$env:PATHSNAP_SCORE_STUDENT='1'
python -m uvicorn pathsnap.api:app --host 0.0.0.0 --port 8000
```

Laptop 2 / frontend terminal:

```powershell
python -m http.server 8080 --bind 0.0.0.0 --directory ui
```

Open http://localhost:8080 on Laptop 2 and set the API address to http://<Laptop-1-LAN-IP>:8000. Keep the receiving API terminal visible. For a one-machine demo use http://127.0.0.1:8000 as the API address. Reserved inference uses PATHSNAP_PARTITION=evaluator_test only once the untouched ten trips are provided; reference files are not read in that partition even if student scoring was enabled.

## Changes and remaining manual work

This audit added verification scripts, independent prediction records, real screenshots, current comparison plots, explicit accuracy weighting and a dependency lock. It enhanced browser checks for a second trip, errors, actual GPS map bounds and loaded tiles. No ML predictor or decoder parameters were changed and no retraining/PDF/publication occurred.

Priority 1: Test the physical LAN and supply the reserved trips under the competition rules. Priority 2: Push all four assigned source contributions; the remote currently lacks parts 2-4. Priority 3: Update the consolidated report with current metrics, plots and genuine screenshots; add team names, repository URL, receiving-terminal screenshot and required AI-assistance disclosure. Each team member must understand and defend their assigned work.
'''
    (VERIFY/'VERIFICATION_REPORT.md').write_text(text,encoding='utf-8')

def build():
    DEST.mkdir(parents=True,exist_ok=True);lock_dependencies();report()
    cfg=json.loads((ROOT/'artifacts/config.json').read_text());fresh=json.loads((VERIFY/'fresh_metrics.json').read_text())
    assert cfg['feature_schema_version']==3 and len(cfg['feature_names'])==12
    current_model_hash=hashlib.sha256((ROOT/'artifacts/model.joblib').read_bytes()).hexdigest()
    assert current_model_hash==fresh['model_sha256'],'Weights changed since verification'
    roles={1:{},2:{},3:{},4:{}}
    def add(role,path,data=None):roles[role][path]=(ROOT/path).read_bytes() if data is None else data
    core={1:['__init__.py','data.py','features.py','alignment.py','phase1.py','baseline.py'],2:['train.py','experiment.py'],3:['decoder.py','calibrate.py','maps.py','reconstruct.py','diagnose.py','report.py'],4:['api.py','verify.py','audit.py','audit_http.py']}
    for role,names in core.items():
        for name in names:add(role,'pathsnap/'+name)
    assert {p.name for p in (ROOT/'pathsnap').glob('*.py')}=={name for names in core.values() for name in names}
    for name in ['.gitignore','README.md','README (4).md','FEATURES.md','TEAM_SPLIT.md','requirements.txt','requirements-lock.txt']:add(1,name)
    for path in (ROOT/'tests').glob('*.py'):
        role=1 if path.name in ['test_data_features.py','test_alignment.py'] else 2 if path.name=='test_training_policy.py' else 3 if path.name=='test_reconstruct.py' else 4
        add(role,path.relative_to(ROOT).as_posix())
    for name in ['inventory.json','eda.png','environment.txt']:
        if name.endswith('.txt'):(ROOT/'results'/name).write_text(decode(ROOT/'results'/name),encoding='utf-8',newline='\n')
        add(1,'results/'+name,decode(ROOT/'results'/name).encode() if name.endswith('.txt') else None)
    for name in ['model.joblib','config.json','feature_schema.json']:add(2,'artifacts/'+name)
    if (ROOT/'retraining_strict.log').exists():
        (ROOT/'results/retrained/training_execution.txt').write_text(decode(ROOT/'retraining_strict.log'),encoding='utf-8',newline='\n')
    if (ROOT/'retraining.log').exists():
        (ROOT/'results/retrained/initial_training_provenance.txt').write_text('Historical first aligned run; its HGB internal split was subsequently disabled. Current authoritative weights/config are artifacts/model.joblib and config.json.\n'+decode(ROOT/'retraining.log'),encoding='utf-8',newline='\n')
    for path in (ROOT/'results/retrained').iterdir():
        if not path.is_file():continue
        role=1 if path.name in ['alignment_training.json','all20_feature_audit.json'] else 3 if path.name in ['decoder_validation.json','weak_trip_diagnostics.json','predicted_route.html'] else 4 if path.name=='browser_e2e.json' else 2
        add(role,path.relative_to(ROOT).as_posix())
    for folder in ['ui','results/final_verification']:
        for path in (ROOT/folder).rglob('*'):
            if path.is_file():
                if path.name=='remote_files.txt':path.write_text(decode(path),encoding='utf-8',newline='\n')
                add(4,path.relative_to(ROOT).as_posix())
    add(4,'package_current.py')
    names={1:'01_DATA_FEATURES_CURRENT',2:'02_MODELS_TRAINING_CURRENT',3:'03_HMM_ROUTES_MAPS_CURRENT',4:'04_API_UI_VERIFICATION_CURRENT'}
    branches={1:'feature/data-features-v3',2:'feature/models-v3',3:'feature/hmm-routes-v3',4:'feature/api-ui-v3'}
    combined={};index={}
    for role,files in roles.items():
        for path,data in files.items():
            assert path not in combined,('Duplicate owner',path)
            combined[path]=data
        staged=' '.join('"'+p+'"' for p in sorted(files))
        instruction=f'''# CURRENT VERIFIED Part {role}

This is the UPDATED schema-v3 project. The extractor validates 20 features; the final Bayesian HistGradientBoosting uses 12. Mean per-trip HMM accuracy is 90.60%; pooled accuracy is 88.89%. Neither is 95%. Only Part 2 contains the current trained model; its SHA-256 is {current_model_hash}.

1. Clone or pull the team's repository: https://github.com/AkashRamesh-19/PT-052-Pathsnap-.git
2. Create your branch: {branches[role]}. Part 1 has an older version on main; its owner must merge the v3 foundation first. Parts 2 and 3 depend on each other's modules for execution; Part 4 integrates last. Each person can push their branch earlier, but run integration checks only after dependencies merge.
3. Compare payload/ paths against the latest repository. Merge existing teammate edits rather than overwrite them. Copy only the CONTENTS of payload/ into the cloned repo root, preserving paths. Do not copy ZIP wrappers or initialize an unrelated repository.
4. Install with python -m pip install -r requirements-lock.txt after Part 1 is merged. On a fully integrated working tree, run python -m unittest discover -s tests -v (23 tests). Official raw data must be obtained separately for live inference; no reserved data is included.

From your clean cloned repository:

```powershell
git switch -c {branches[role]}
# Copy reviewed payload contents here, preserving folder paths.
git status --short
git add -- {staged}
git diff --cached --check
git diff --cached --stat
git commit -m "Add verified current Part {role}"
git push -u origin {branches[role]}
```

Open a pull request and coordinate merges. Do not force-push and do not push someone else's whole-project payload as your own part. Existing experiments and code were prepared locally with Codex assistance, not independently performed by each ZIP recipient. Review and understand your assigned work and disclose assistance under course policy. Team members must be able to explain it.

The model/results are already trained and verified; do not rerun expensive training merely to push files. Inference and tests require the integrated project. Full verification report is supplied in Part 4; shared startup instructions are in README.md and START_HERE_CURRENT.md. Historical eight-feature model weights, stale PDFs, raw datasets, caches, virtual environments, secrets and Git history are excluded.
'''
        records={p:hashlib.sha256(data).hexdigest() for p,data in sorted(files.items())}
        target=DEST/(names[role]+'.zip')
        with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
            for p,data in sorted(files.items()):z.writestr('payload/'+p,data)
            z.writestr('HOW_TO_PUSH.md',instruction.encode());z.writestr('FILE_HASHES.json',json.dumps(records,indent=2))
        with zipfile.ZipFile(target) as z:
            assert z.testzip() is None
            for p,digest in records.items():assert hashlib.sha256(z.read('payload/'+p)).hexdigest()==digest
        index[str(role)]=dict(zip=target.name,branch=branches[role],files=len(files),bytes=target.stat().st_size)
    # Validate the exact combined ZIP payload in a new cache checkout.
    for path,data in combined.items():
        assert (ROOT/path).read_bytes()==data,('Package differs from current workspace',path)
    root=ROOT/'data_cache';validation=Path(tempfile.mkdtemp(prefix='verified-current-union-',dir=root)).resolve();assert validation.is_relative_to(root.resolve())
    for path,data in combined.items():
        dest=validation/path;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
    run=subprocess.run([__import__('sys').executable,'-m','unittest','discover','-s','tests','-v'],cwd=validation,capture_output=True,text=True)
    (DEST/'combined_package_tests.txt').write_text(run.stdout+run.stderr,encoding='utf-8');assert run.returncode==0
    count=int(re.search(r'Ran (\d+) tests',run.stdout+run.stderr).group(1));assert count==23
    with zipfile.ZipFile(DEST/'PathSnap_COMPLETE_CURRENT.zip','w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p,data in sorted(combined.items()):z.writestr('PathSnap/'+p,data)
    intro=f'''# Current verified PathSnap contributions

USE THESE FILES, NOT THE EARLIER ZIPs IN THE PARENT DIRECTORY.

1. Your update: {names[1]}.zip
2. Models teammate: {names[2]}.zip
3. HMM/maps teammate: {names[3]}.zip
4. API/UI/QA teammate: {names[4]}.zip

Each ZIP contains payload/ with repository-relative files plus HOW_TO_PUSH.md and FILE_HASHES.json. Push only your assigned payload on its own branch. The team repository currently has only the older Part 1. Integrate the new Part 1 first, then both Parts 2 and 3, then Part 4. Do not upload ZIPs as source and do not overwrite teammate changes.

PathSnap_COMPLETE_CURRENT.zip is the combined project for running locally, not one person's contribution. The union of all four role ZIPs passed {count} tests. All payload hashes were compared to current workspace files. Final model SHA-256: {current_model_hash}. Final model: schema v3, 12 actual features, final strict training with early_stopping=False. Old 8-feature models and historical PDF are excluded.

Accuracy uses inferred aligned labels: classifier 87.20% mean / 85.59% pooled; HMM 90.60% mean / 88.89% pooled. Current metrics, plots, real browser screenshots, independent prediction records and verification report are included. Reserved evaluator trips and a physical two-laptop run remain unverified. No models were retrained, no PDF generated, and nothing pushed while packaging.

To run the combined project, extract PathSnap_COMPLETE_CURRENT.zip, enter PathSnap/, install requirements-lock.txt, and follow results/final_verification/VERIFICATION_REPORT.md. Raw official train/ and student_test/ data must be supplied separately; the API cannot run those trips without it. Model artifacts should only be loaded from this trusted team archive.
'''
    (DEST/'START_HERE_CURRENT.md').write_text(intro,encoding='utf-8')
    save=dict(feature_schema_version=3,final_feature_count=12,model_sha256=current_model_hash,source='current verified workspace, not earlier contribution archives',combined_tests=count,parts=index)
    (DEST/'CURRENT_PACKAGE_INDEX.json').write_text(json.dumps(save,indent=2),encoding='utf-8')
    with zipfile.ZipFile(DEST/'ALL_FOUR_CURRENT.zip','w',zipfile.ZIP_DEFLATED) as z:
        for name in [*(r['zip'] for r in index.values()),'START_HERE_CURRENT.md','CURRENT_PACKAGE_INDEX.json','combined_package_tests.txt']:z.write(DEST/name,name)
    for path in DEST.glob('*.zip'):
        with zipfile.ZipFile(path) as z:assert z.testzip() is None
    print(json.dumps(save,indent=2));print('Verified output:',DEST)

if __name__=='__main__':build()
