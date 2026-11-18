"""Real HTTP checks against the already running student-development API."""
import argparse
import json
import time
import httpx
import numpy as np
from .audit import OUT
from .data import ROOT
from .train import save

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--url',default='http://127.0.0.1:8000');args=parser.parse_args()
    rows=[];logs=[]
    with httpx.Client(base_url=args.url,timeout=180) as client:
        for path in ('/health','/trips','/','/docs','/openapi.json'):
            response=client.get(path);assert response.status_code==200,(path,response.status_code)
            logs.append(f'GET {path} -> {response.status_code}')
        health=client.get('/health').json();assert health['model_ready'] and health['feature_count']>=12
        listing=client.get('/trips').json();assert listing['partition']=='student_test','Audit must not run against reserved evaluator data'
        assert listing['ready'] and len(listing['trips'])==10
        for trip in listing['trips']:
            start=time.perf_counter();response=client.post('/predict/'+trip);assert response.status_code==200,response.text[:250]
            data=response.json();arrays=np.load(OUT/'predictions'/f'{trip}.npz')
            np.testing.assert_array_equal(data['predicted_arc_ids'],arrays['hmm_arc_ids'])
            assert data['observations']==len(arrays['hmm_arc_ids'])
            assert 'reference_arc_ids' not in data and 'Reference route (development only)' not in data['map_html']
            folder=ROOT/'student_test'/trip;arcs=np.loadtxt(next(folder.glob('*.arcs')),dtype=int,ndmin=2)
            for section in data['route_sections']:
                if len(section)>1:assert np.array_equal(arcs[section[:-1],1],arcs[section[1:],0])
            score=float(np.mean(arrays['hmm_arc_ids']==arrays['aligned_proxy_arc_ids']))
            if 'proxy_pointwise_accuracy' in data['metrics']:assert abs(score-data['metrics']['proxy_pointwise_accuracy'])<1e-12
            row=dict(trip=trip,http_status=response.status_code,observations=data['observations'],seconds=time.perf_counter()-start,score=score,first_predicted_arcs=data['predicted_arc_ids'][:12],route_sections=len(data['route_sections']),metrics=data['metrics'])
            rows.append(row);logs.append(f'POST /predict/{trip} -> 200; observations={row["observations"]}; matches independently recalculated states; seconds={row["seconds"]:.2f}')
            print(logs[-1],flush=True)
        for path,method,expected in [('/predict/not-a-trip','POST',404),('/predict/99999999','POST',404),('/predict/'+listing['trips'][0],'GET',405)]:
            response=client.request(method,path);assert response.status_code==expected,(path,response.status_code)
            logs.append(f'{method} {path} -> {response.status_code} (expected)')
        response=client.options('/predict/'+listing['trips'][0],headers={'Origin':'http://127.0.0.1:8080','Access-Control-Request-Method':'POST','Access-Control-Request-Headers':'content-type'})
        assert response.status_code==200 and response.headers.get('access-control-allow-origin') in ('*','http://127.0.0.1:8080')
        logs.append('Cross-origin POST preflight -> 200, allowed')
    save(OUT/'http.json',dict(status='PASS',health=health,student_trips=rows,checks=logs,cors=True,reserved_trips_executed=False))
    (OUT/'http_requests.txt').write_text('\n'.join(logs)+'\n',encoding='utf-8')
    print('HTTP audit PASS: all 10 student trips, endpoint errors and CORS',flush=True)

if __name__=='__main__':main()
