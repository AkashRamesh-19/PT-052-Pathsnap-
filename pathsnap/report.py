"""Build honest plots and a consolidated PDF directly from recorded results."""
import json
import os
import textwrap
from pathlib import Path
from .data import ROOT
cache=ROOT/'data_cache'/'matplotlib'
cache.mkdir(parents=True,exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR',str(cache))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def generate():
    out=ROOT/'results'; reports=ROOT/'reports'; reports.mkdir(exist_ok=True)
    rows=json.loads((out/'comparison.json').read_text())
    config=json.loads((ROOT/'artifacts/config.json').read_text())
    if config.get('feature_schema_version',0)>=3:
        raise RuntimeError('Current pipeline uses results/retrained JSON metrics; legacy PDF generation would mix incompatible experiments')
    inv=json.loads((out/'inventory.json').read_text())
    trials=json.loads((out/'bayesian_trials.json').read_text())
    with PdfPages(reports/'technical_report.pdf') as pdf:
        def page(title, paragraphs):
            fig=plt.figure(figsize=(8.27,11.69)); fig.text(.08,.94,title,fontsize=17,weight='bold'); y=.88
            for paragraph in paragraphs:
                lines=[]
                for line in str(paragraph).splitlines():
                    lines.extend(textwrap.wrap(line,width=93) or [''])
                fig.text(.08,y,'\n'.join(lines),fontsize=10,va='top',linespacing=1.5)
                y-=len(lines)*.019+.025
            pdf.savefig(fig); plt.close(fig)
        page('PathSnap | Technical report',[
            'UE24CS352A - Machine Learning, PES University. Generated from actual local experiment artifacts. Team names and public repository URL must be supplied by the team.',
            f'Dataset: {sum(r.get("partition")=="train" for r in inv)} training and {sum(r.get("partition")=="student_test" for r in inv)} student-testing trips. Evaluator data was not accessed. Seed: 352. Three-fold GroupKFold holds out complete trips. Up to '+str(config['samples_per_trip'])+' evenly spaced observations per training trip are used for feasible local experiments; all student observations are evaluated.',
            'Label limitation: route files list traversed arc IDs without observation alignment. Supervision and pointwise evaluation use the closest segment belonging to the supplied reference route. These are geometric proxy labels. Repeated visits, intersections, opposite-direction overlapping arcs and route inaccuracies can make those labels ambiguous. Reported accuracy is not verified observation-level ground-truth accuracy.',
            'AI assistance disclosure: implementation and this generated report were produced with Codex assistance. The team must review the code and experiments, understand and defend the approach, and follow the event disclosure rules.',
            'Submission status: the software and development experiments are local. Reserved evaluator data, two-laptop demonstration screenshots, team information and public GitHub publication remain external submission tasks. No screenshots or measured results have been fabricated.'
        ])
        valid=[r for r in inv if 'error' not in r]
        fig,axes=plt.subplots(1,2,figsize=(10,5))
        for partition,color in [('train','#376eb4'),('student_test','#20a47a')]:
            subset=[r for r in valid if r['partition']==partition]
            axes[0].hist([r['observations'] for r in subset],bins=12,alpha=.6,label=partition,color=color)
            axes[1].scatter([r['nodes'] for r in subset],[r['arcs'] for r in subset],label=partition,color=color,s=12)
        axes[0].set(xlabel='GPS observations per trip',ylabel='Trips',title='Development trajectory sizes');axes[1].set(xlabel='Nodes',ylabel='Directed arcs',title='Network sizes');axes[0].legend();axes[1].legend();fig.tight_layout();fig.savefig(out/'eda.png',dpi=150);pdf.savefig(fig);plt.close(fig)
        page('1. Dataset and feature engineering',[
            f'Observed GPS counts range from {min(r["observations"] for r in valid)} to {max(r["observations"] for r in valid)}. Largest time gap: {max(r["max_gap"] for r in valid):.1f} seconds. Inventory includes each file size, coordinate extent, network counts, weak components and isolated nodes. Invalid inputs fail validation rather than being silently discarded. See results/inventory.json.',
            'Coordinates are longitude/latitude. A local equirectangular projection uses the network mean latitude, with 111320*cos(latitude) metres per longitude degree and 111320 metres per latitude degree. Appropriate for these local networks; projection distortion should be reconsidered for wide-area datasets.',
            f'Six exact nearest line-segment candidates per observation are found with a midpoint KD-tree and a maximum segment half-length bound. This artifact uses {len(config["feature_names"])} features: '+', '.join(config['feature_names'])+'. Geometry and motion features transfer across trip-specific arc IDs. Schema-v2 definitions and units are documented in FEATURES.md; eight-feature artifacts retain the older schema.',
            'Reference-route files are opened only for training targets or after inference for scoring. They are never prediction features. Scaling is fit inside each training fold. Train/student trip IDs are checked for overlap. Evaluator data is excluded from exploration, fitting, model selection and tuning.',
            'All candidate rows from a trip share one cross-validation group. Candidate binary Macro F1 is the CV metric. Model-family selection uses student-trip mean observation Top-1 proxy accuracy. Final tuned-versus-untuned configuration is selected by grouped CV Macro F1. Candidate misses count in observation accuracy.'
        ])
        gaps=sorted(valid,key=lambda r:r['max_gap'],reverse=True)[:5]
        page('1. Observed data-quality findings',[
            f'Total development observations: {sum(r["observations"] for r in valid):,}. Every supplied trip passed finite-value, shape, coordinate-range, timestamp-ordering and arc-index validation. A zero missing-value count describes these files; it is not evidence that GPS measurements are noise-free.',
            'Largest observed gaps (partition/trip/seconds): '+', '.join(f'{r["partition"]}/{r["trip"]}/{r["max_gap"]:.0f}' for r in gaps)+'. These gaps justify explicit sequence resets rather than assuming continuous movement.',
            f'{sum(r["weak_components"]>1 for r in valid)} trip networks contain multiple weak components; {sum(r["isolated_nodes"]>0 for r in valid)} contain isolated nodes. Connectivity does not guarantee that directed travel between any two segments is possible. The decoder therefore checks directed paths instead of relying on weak-component membership.',
            'Feature choice follows the observed task structure: raw arc IDs vary by network; distance and heading express reusable geometric relationships. Irregular time steps require speed to use actual elapsed seconds. Reference-route membership alone cannot align observations chronologically, so all pointwise conclusions are limited to geometric proxies.'
        ])
        rationale={
            'LogisticRegression':('linear probabilistic','Logistic loss; scaled features, C=1, max_iter=400. Fast calibrated baseline; limited nonlinear interactions.'),
            'GaussianNB':('probabilistic generative','Gaussian likelihood; var_smoothing=1e-8 after scaling. Fast, but correlated geometry features violate independence.'),
            'DecisionTree':('single tree','Gini impurity; depth=10, leaf minimum=15. Learns nonlinear thresholds; prone to overfitting.'),
            'RandomForest':('bagged tree ensemble','Gini impurity; 80 trees, depth=14, leaf minimum=5. Robust interactions with higher computation and memory.'),
            'HistGradientBoosting':('boosting','Binary log loss; 100 iterations, 15 leaves, L2=1. Learns nonlinear combinations; probability quality and regularisation matter.')}
        for row in rows:
            name=row['model']; family=name.replace('_tuned',''); info=rationale[family]
            page('2. Model: '+name,[f'Family: {info[0]}. {info[1]}', 'Actual configuration: '+row['configuration'], f'Grouped CV candidate Macro F1: {row["cv_macro_f1"]:.4f}. Training Macro F1: '+(f'{row["train_macro_f1"]:.4f}. CV standard deviation: {row["cv_std"]:.4f}. A positive train/CV gap indicates possible overfitting; this is not independent physical ground-truth evidence.' if 'train_macro_f1' in row else 'not separately recorded for tuned model.'), '\n'.join(f'{k}: {row[k]:.6f}' for k in ['top1_accuracy','macro_precision','macro_recall','macro_f1','log_loss','candidate_miss_rate','classifier_continuity','route_continuity','hmm_accuracy','sequence_exact_match','training_seconds','inference_seconds']), 'Metrics are unweighted means across the ten student trips. Macro precision/recall/F1 and log loss refer to binary candidate relevance, not a shared global arc-ID classification space. Inference time includes features and Viterbi, excludes network loading, reference scoring and maps. Per-trip files include confusion matrices.'])
        labels=[r['model'] for r in rows]
        fig,ax=plt.subplots(figsize=(11.7,8.3));ax.axis('off');ax.set_title('Model comparison | measured development results',pad=25)
        columns=['Model','CV F1','Top-1','HMM','Macro F1','Log loss','Continuity','Seconds']
        values=[[r['model']]+[f'{r[k]:.4f}' for k in ['cv_macro_f1','top1_accuracy','hmm_accuracy','macro_f1','log_loss','route_continuity','inference_seconds']] for r in rows]
        table=ax.table(cellText=values,colLabels=columns,loc='center',cellLoc='center',colWidths=[.26]+[.105]*7);table.auto_set_font_size(False);table.set_fontsize(8);table.scale(1,2.5);fig.tight_layout();pdf.savefig(fig);plt.close(fig)
        for keys,title in [(['cv_macro_f1'],'Grouped CV candidate Macro F1'),(['top1_accuracy','macro_f1','hmm_accuracy'],'Student-test performance'),(['log_loss'],'Student-test candidate Log Loss')]:
            fig,ax=plt.subplots(figsize=(10,6)); import numpy as np
            pos=np.arange(len(rows)); width=.8/len(keys)
            for j,key in enumerate(keys): ax.bar(pos+j*width,[r[key] for r in rows],width,label=key)
            ax.set_xticks(pos+.4-width/2,labels,rotation=25,ha='right'); ax.set_title(title); ax.legend(); fig.tight_layout(); fig.savefig(out/(keys[0]+'.png'),dpi=150); pdf.savefig(fig); plt.close(fig)
        page('3. Bayesian hyperparameter optimisation',[
            'Gaussian-process surrogate with Matern kernel plus white noise. First three trials use seeded random initial points; subsequent trials maximise expected improvement over 256 seeded proposals. Objective: mean binary candidate Macro F1 from three-fold training-trip GroupKFold. This is Bayesian optimisation, not random search alone.',
            'Search spaces: LogisticRegression C log-uniform [0.001,100]; GaussianNB smoothing log-uniform [1e-12,1e-3]; DecisionTree depth integer [4,20], leaf minimum [3,40]; RandomForest depth [6,22], leaf minimum [2,20], 80 trees; HistGradientBoosting learning rate log-uniform [0.01,0.2], leaves [7,32], 100 iterations. Only the selected family is searched. The second unit coordinate is unused for one-parameter families.',
            f'Trials: {len(trials)}. Total recorded trial time: {sum(t["seconds"] for t in trials):.1f} seconds. Best recorded trial: '+str(max(trials,key=lambda t:t['cv_macro_f1'])),
            f'Final selected artifact: {config["model"]}; tuned={config["tuned"]}. The comparison plots show tuned and untuned measured student-test results. Results/bayesian_trials.json retains every trial.'
        ])
        page('4. HMM and Viterbi reconstruction',[
            'Hidden state: a candidate directed arc at each GPS observation. Emissions are positive binary classifier probabilities normalised across that observation\'s candidates. The decoder maximises log emission plus transition potentials; these potentials form an unnormalised HMM-style sequence model.',
            'Transition log potential = -abs(directed road distance - GPS displacement)/beta. Road distance includes the remaining fraction of the previous arc, a shortest path between its end node and the next arc start, and the fraction of the next arc. Forward movement on the same arc uses only the fraction difference. A heap-based bounded Dijkstra search handles the directed network.',
            f'Decoder configuration: {config["decoder"]}. Beta is selected from 30, 60 and 120 metres using student-test HMM proxy accuracy, with continuity as a tie-breaker; see results/decoder_search.json. Search distance is bounded by speed*gap+100m, between 200m and 10km. Transitions outside that bound are invalid. Gaps over 60 seconds reset the sequence. If all paths become invalid, the decoder starts a new independent section and reports a reset.',
            'Viterbi recurrence: score[t,j] = log emission[t,j] + max_i(score[t-1,i] + log transition[i,j]); predecessor indices are retained and backtracked. Classifier-only and HMM pointwise proxy accuracy and continuity are recorded for every model. Exact sequence match is strict equality of all observation proxy labels.',
            'Speed and gap defaults are engineering choices. Beta calibration on student testing is development evidence and may overfit those ten trips. Continuity is directed reachability within the movement bound; it does not mean immediate arc adjacency. Predicted maps render selected road segments separately, so resets do not create fictitious connector lines. Geometry of intervening shortest-path arcs is not included in the API output.'
        ])
        if (out/'final_metrics.json').exists():
            final=json.loads((out/'final_metrics.json').read_text())
            page('4. Final decoder evidence',[
                f'Chosen beta: {config["decoder"]["beta"]:.0f}m. Classifier Top-1 proxy accuracy: {final["top1_accuracy"]:.2%}; HMM proxy accuracy: {final["hmm_accuracy"]:.2%}. Classifier continuity: {final["classifier_continuity"]:.2%}; HMM continuity: {final["route_continuity"]:.2%}. Candidate miss rate: {final["candidate_miss_rate"]:.2%}. Exact sequence-match rate: {final["sequence_exact_match"]:.2%}.',
                'The measured decoder improves directed continuity but reduces geometric-proxy pointwise accuracy on average. The two metrics reflect different objectives. A continuity potential can overrule a locally preferred segment; geometric proxy labels also cannot reliably resolve opposite directed arcs and repeated traversals. These explanations are plausible mechanisms, not independently verified causal findings.',
                'The complete classifier-only and HMM metrics are retained for each trip, including worse cases. No claim is made that sequential smoothing always improves accuracy. The final decoder remains included to provide the required coherent-road sequence stage, with its tradeoff visible to reviewers.'
            ])
        # Static geographical view from the same actual prediction for the PDF.
        import joblib
        from .api import infer
        from .data import read_reference
        folder=next(p for p in sorted((ROOT/'student_test').iterdir()) if p.is_dir())
        trip,net,predicted,_=infer(folder)
        reference=read_reference(folder,len(trip.arcs))
        from matplotlib.collections import LineCollection
        fig,ax=plt.subplots(figsize=(9,7))
        ax.add_collection(LineCollection(trip.nodes[trip.arcs[np.unique(reference)]],colors='#e6a42d',linewidths=3,label='Reference'))
        ax.add_collection(LineCollection(trip.nodes[trip.arcs[np.unique(predicted)]],colors='#20a47a',linewidths=1.5,label='Prediction'))
        ax.scatter(trip.track[:,0],trip.track[:,1],s=2,c='#2196f3',label='GPS'); ax.autoscale(); ax.set_aspect(1/np.cos(np.deg2rad(trip.track[:,1].mean()))); ax.legend(); ax.set_title(f'5. Actual trip {trip.id}: geographical overview'); ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude'); fig.tight_layout(); fig.savefig(out/'route_overview.png',dpi=150); pdf.savefig(fig);plt.close(fig)
        page('5-6. Maps, API, UI and deployment',[
            'Interactive Folium output: results/representative_map.html, generated from actual observations, road geometry, reference route and model predictions. Layer controls, arc tooltips and a colour legend are included. The preceding page is a static plot of the same actual route, not a screenshot of the Folium browser map. A browser screenshot of the Folium view is still required for the final submission.',
            'Laptop 1: python -m uvicorn pathsnap.api:app --host 0.0.0.0 --port 8000. Laptop 2: python -m http.server 8080 --directory ui. Open http://localhost:8080 and enter Laptop 1\'s LAN address. The UI lists server trips, makes a POST request and displays returned metrics and an interactive map.',
            'Endpoints: GET /health, GET /trips, POST /predict/{trip_id}. Model and preprocessing load from a trusted local joblib artifact. The API logs trip IDs, timing, observation counts, reset counts and aggregate metrics. Only prediction IDs and aggregate scores leave the server; reference segments are omitted from API maps.',
            'Development defaults to student_test. Set PATHSNAP_PARTITION=evaluator_test on Laptop 1 only after the untouched ten reserved trips are installed. Do not tune against those results. Missing model artifacts produce HTTP 503; invalid/missing trips and malformed data produce explicit errors.',
            'Final submission still requires team details, public repository publication, a real two-laptop demonstration, browser map screenshot, UI screenshot and receiving-terminal screenshot. These cannot be represented as completed without the corresponding evidence.'
        ])
    import csv
    keys=[k for k in rows[0] if k!='configuration']
    with (out/'comparison.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    summary=['<!-- measured-results:start -->','## Measured local results','',
             'Actual development run: 80 training trips, 10 student-test trips, up to '+str(config['samples_per_trip'])+' training observations per trip, three-fold trip-grouped CV and '+str(len(trials))+' Bayesian trials. All accuracy values below use the documented geometric proxy labels. Values are means across student trips.','',
             '| Model | CV candidate Macro F1 | Student Top-1 | Student candidate Macro F1 | Log loss | HMM accuracy | HMM continuity | Inference seconds/trip |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        summary.append('| '+r['model']+' | '+' | '.join(f'{r[k]:.4f}' for k in ['cv_macro_f1','top1_accuracy','macro_f1','log_loss','hmm_accuracy','route_continuity','inference_seconds'])+' |')
    summary+=['',f'Final model: **{config["model"]}**, tuned configuration selected: **{config["tuned"]}**. Model-comparison decoding uses beta=60m; the final beta calibration is separate.']
    if (out/'final_metrics.json').exists():
        final=json.loads((out/'final_metrics.json').read_text())
        summary+=['',f'Final decoder beta: **{config["decoder"]["beta"]:.0f}m**. Final classifier Top-1: **{final["top1_accuracy"]:.2%}**; HMM proxy accuracy: **{final["hmm_accuracy"]:.2%}**; classifier continuity: **{final["classifier_continuity"]:.2%}**; HMM continuity: **{final["route_continuity"]:.2%}**.']
    summary+=['','These are development measurements, not reserved-evaluator results. See the PDF for model-specific configurations, limitations and interpretation.','<!-- measured-results:end -->']
    readme=ROOT/'README.md';text=readme.read_text(encoding='utf-8');start='<!-- measured-results:start -->';end='<!-- measured-results:end -->'
    if start in text:
        a=text.index(start);b=text.index(end,a)+len(end);text=text[:a]+'\n'.join(summary)+text[b:]
    else: text+='\n\n'+'\n'.join(summary)+'\n'
    readme.write_text(text,encoding='utf-8')

if __name__=='__main__': generate()
