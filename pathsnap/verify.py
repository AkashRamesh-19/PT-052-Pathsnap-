"""Exercise every available trip through FastAPI, without exporting references."""
import json
import argparse
import httpx
from .api import app
from .data import ROOT

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--url',help='Exercise a running API over HTTP instead of in-process')
    args=parser.parse_args()
    rows=[]
    if args.url:
        session=httpx.Client(base_url=args.url,timeout=300)
    else:
        from fastapi.testclient import TestClient
        session=TestClient(app)
    with session as client:
        assert client.get('/').status_code==200
        assert client.get('/health').json()['model_ready']
        listing=client.get('/trips').json()
        assert listing['ready'],f'Need ten trips: {listing}'
        for trip_id in listing['trips']:
            response=client.post('/predict/'+trip_id)
            assert response.status_code==200, response.text[:300]
            data=response.json()
            assert len(data['predicted_arc_ids'])==data['observations']
            assert 'reference_arc_ids' not in data
            assert 'Reference route (development only)' not in data['map_html']
            row={k:data[k] for k in ('trip_id','partition','observations','seconds','metrics')}
            rows.append(row)
            print(json.dumps(row),flush=True)
    target=ROOT/'results'/('api_verification_'+listing['partition']+'.json')
    target.write_text(json.dumps(rows,indent=2),encoding='utf-8')
    print(f'All {len(rows)} API trip requests passed',flush=True)

if __name__=='__main__': main()
