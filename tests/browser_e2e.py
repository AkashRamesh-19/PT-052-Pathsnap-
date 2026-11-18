"""Real installed-Chromium UI test over CDP; no mocked API or screenshots."""
import argparse
import base64
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
from urllib.request import urlopen
import websocket

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--ui',default='http://127.0.0.1:8080');parser.add_argument('--api',default='http://127.0.0.1:8000')
    parser.add_argument('--trips',type=int,default=1);parser.add_argument('--screenshots',action='store_true');parser.add_argument('--output',default='results/retrained');args=parser.parse_args()
    output=(ROOT/args.output).resolve()
    if not output.is_relative_to(ROOT.resolve()) or not 1<=args.trips<=10:parser.error('Use a workspace output directory and 1-10 trips')
    output.mkdir(parents=True,exist_ok=True)
    browser=next((p for p in [Path('C:/Program Files/Google/Chrome/Application/chrome.exe'),Path('C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe')] if p.exists()),None)
    if browser is None:raise RuntimeError('No installed Chrome/Edge: browser rendering not verified')
    base=ROOT/'data_cache/browser-tests';base.mkdir(parents=True,exist_ok=True)
    profile=Path(tempfile.mkdtemp(dir=base)).resolve()
    assert profile.is_relative_to(base.resolve())
    process=subprocess.Popen([str(browser),'--headless=new','--disable-gpu','--no-first-run','--no-default-browser-check','--remote-debugging-port=9223','--remote-allow-origins=http://127.0.0.1:9223',f'--user-data-dir={profile}','about:blank'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    connection=None;counter=0;errors=[];requests=[]
    try:
        deadline=time.monotonic()+30
        while True:
            try:
                targets=json.load(urlopen('http://127.0.0.1:9223/json',timeout=2));target=next(t for t in targets if t['type']=='page');break
            except Exception:
                if time.monotonic()>deadline:raise
                time.sleep(.3)
        connection=websocket.create_connection(target['webSocketDebuggerUrl'],origin='http://127.0.0.1:9223',timeout=15)
        def call(method,params=None):
            nonlocal counter
            counter+=1;wanted=counter;connection.send(json.dumps(dict(id=wanted,method=method,params=params or {})))
            while True:
                response=json.loads(connection.recv())
                if response.get('method')=='Runtime.exceptionThrown':errors.append(response['params']['exceptionDetails'].get('text','JS exception'))
                if response.get('method')=='Network.requestWillBeSent':
                    request=response['params']['request']
                    if request['url'].startswith(args.api+'/predict/'):
                        requests.append(dict(url=request['url'],method=request['method']))
                if response.get('id')==wanted:
                    if 'error' in response:raise RuntimeError(response['error'])
                    return response.get('result',{})
        def evaluate(expression):
            value=call('Runtime.evaluate',dict(expression=expression,returnByValue=True,awaitPromise=True))
            if 'exceptionDetails' in value:raise RuntimeError(value['exceptionDetails'])
            return value.get('result',{}).get('value')
        def wait(expression,seconds=90):
            deadline=time.monotonic()+seconds
            while not evaluate(expression):
                if time.monotonic()>deadline:raise TimeoutError(expression+'; JS errors: '+str(errors))
                time.sleep(.3)
        call('Page.enable');call('Runtime.enable');call('Network.enable')
        call('Emulation.setDeviceMetricsOverride',dict(width=1400,height=1000,deviceScaleFactor=1,mobile=False))
        call('Page.navigate',dict(url=args.ui));wait("document.readyState==='complete' && !!document.getElementById('api')")
        wait("!document.getElementById('connect').disabled")
        evaluate("document.getElementById('api').value='not-a-valid-url';document.getElementById('connect').click()")
        wait("!document.getElementById('connect').disabled && document.getElementById('run').disabled && document.getElementById('status').textContent.length>0")
        evaluate("document.getElementById('api').value="+json.dumps(args.api)+";document.getElementById('connect').click()")
        wait("document.getElementById('trip').options.length===10 && !document.getElementById('run').disabled")
        def screenshot(name,clip=None):
            options=dict(format='png',captureBeyondViewport=True,fromSurface=True)
            if clip:options['clip']=clip
            path=output/name;path.write_bytes(base64.b64decode(call('Page.captureScreenshot',options)['data']))
            return path.relative_to(ROOT).as_posix()
        trips=[];screenshots=[]
        for i in range(args.trips):
            evaluate(f"document.getElementById('trip').selectedIndex={i};document.getElementById('run').click()")
            wait("document.getElementById('status').textContent.includes('Completed trip')",seconds=240)
            wait("(()=>{const f=document.getElementById('map');const id=f.srcdoc.match(/id=\"(map_[^\"]+)\"/)?.[1];return !f.hidden && !!id && !!f.contentDocument?.getElementById(id)?.classList.contains('leaflet-container') && f.contentDocument.querySelectorAll('path.leaflet-interactive').length>0})()",seconds=90)
            visible=evaluate("(()=>{const d=document.getElementById('map').contentDocument;const label=[...d.querySelectorAll('.leaflet-control-layers label')].find(x=>x.textContent.includes('Relevant road network'));if(!label)return false;const input=label.querySelector('input');if(!input.checked)input.click();return input.checked})()")
            assert visible,'Road-network layer missing'
            item=evaluate("({trip:document.getElementById('trip').value,status:document.getElementById('status').textContent,observations:document.getElementById('count').textContent,leaflet_paths:document.getElementById('map').contentDocument.querySelectorAll('path.leaflet-interactive').length,legend:document.getElementById('map').contentDocument.body.textContent.includes('GPS') && document.getElementById('map').contentDocument.body.textContent.includes('prediction'),map_bounds_available:document.getElementById('map').srcdoc.includes('fitBounds')})")
            wait("(()=>{const f=document.getElementById('map'),w=f.contentWindow,id=f.srcdoc.match(/id=\"(map_[^\"]+)\"/)?.[1],m=w[id];let loading=false;m.eachLayer(l=>{if(l instanceof w.L.TileLayer && l.isLoading())loading=true});return !loading && f.contentDocument.querySelectorAll('img.leaflet-tile-loaded').length>0})()",seconds=45)
            item['actual_bounds_contain_gps']=evaluate("(()=>{const f=document.getElementById('map'),w=f.contentWindow,id=f.srcdoc.match(/id=\"(map_[^\"]+)\"/)?.[1],m=w[id];let gps;m.eachLayer(l=>{if(l instanceof w.L.Polyline && l.options.color==='#2196f3')gps=l});return !!gps && m.getBounds().contains(gps.getBounds())})()")
            item['loaded_basemap_tiles']=evaluate("document.getElementById('map').contentDocument.querySelectorAll('img.leaflet-tile-loaded').length")
            assert item['legend'] and item['map_bounds_available'] and item['actual_bounds_contain_gps'];trips.append(item)
            if args.screenshots:
                time.sleep(1)
                screenshots.append(screenshot(f'ui_{item["trip"]}.png'))
                clip=evaluate("(()=>{const r=document.getElementById('map').getBoundingClientRect();return {x:r.x+scrollX,y:r.y+scrollY,width:r.width,height:r.height,scale:1}})()")
                screenshots.append(screenshot(f'map_{item["trip"]}.png',clip))
        result=dict(trips=trips,error_state_verified=True,screenshots=screenshots)
        call('Emulation.setDeviceMetricsOverride',dict(width=390,height=844,deviceScaleFactor=1,mobile=True))
        time.sleep(.5)
        result['mobile_no_horizontal_overflow']=evaluate('document.documentElement.scrollWidth<=window.innerWidth+2')
        assert result['mobile_no_horizontal_overflow'],result
        assert not errors,errors
        if args.screenshots:result['screenshots'].append(screenshot('mobile_ui.png'))
        assert len(requests)==args.trips and all(r['method']=='POST' for r in requests),requests
        result.update(ui=args.ui,api=args.api,real_browser=True,mocked_api=False,actual_prediction_requests=requests)
        out=output/'browser_e2e.json';out.write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(json.dumps(result,indent=2))
        try:call('Browser.close')
        except (websocket.WebSocketConnectionClosedException,ConnectionResetError):pass
    finally:
        if connection:connection.close()
        try:process.wait(timeout=8)
        except subprocess.TimeoutExpired:process.terminate();process.wait(timeout=8)

if __name__=='__main__':main()
