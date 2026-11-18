import folium
import numpy as np
from .data import ROOT

def make_map(trip, predicted, reference=None, route_sections=None):
    m=folium.Map(location=trip.track[:, :2].mean(0)[::-1].tolist(), tiles='OpenStreetMap')
    road=folium.FeatureGroup(name='Relevant road network',show=False)
    # Display nearby network geometry, rather than hundreds of thousands of arcs.
    low=trip.track[:, :2].min(0)-.003; high=trip.track[:, :2].max(0)+.003
    coords=trip.nodes[trip.arcs]
    relevant=((coords>=low)&(coords<=high)).all(axis=(1,2))
    if relevant.any():
        # One geometry avoids thousands of Folium template objects per request.
        folium.GeoJson({'type':'Feature','properties':{},'geometry':{'type':'MultiLineString','coordinates':coords[relevant].tolist()}},style_function=lambda _:dict(color='#8795a7',weight=1,opacity=.35)).add_to(road)
    road.add_to(m)
    gps=folium.FeatureGroup(name='GPS observations')
    folium.PolyLine(trip.track[:, 1::-1].tolist(),color='#2196f3',weight=2,opacity=.5).add_to(gps)
    for i in range(0,len(trip.track),max(1,len(trip.track)//250)):
        folium.CircleMarker(trip.track[i,1::-1].tolist(),radius=2,color='#2196f3',tooltip=f'GPS {i}').add_to(gps)
    gps.add_to(m)
    route=predicted if route_sections is None else [arc for section in route_sections for arc in section]
    for name, ids, color, weight in [('Predicted segments',route,'#20b486',5),('Reference route (development only)',reference,'#f5a623',3)]:
        if ids is None: continue
        layer=folium.FeatureGroup(name=name)
        for arc in dict.fromkeys(int(a) for a in ids):
            folium.PolyLine(trip.nodes[trip.arcs[arc]][:,::-1].tolist(),color=color,weight=weight,opacity=.8,tooltip=f'Arc {arc}').add_to(layer)
        layer.add_to(m)
    m.fit_bounds([trip.track[:,1::-1].min(0).tolist(),trip.track[:,1::-1].max(0).tolist()])
    folium.LayerControl().add_to(m)
    reference_legend=' · <span style="color:#f5a623">reference (development only)</span>' if reference is not None else ''
    m.get_root().html.add_child(folium.Element('<div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:white;padding:12px;font:14px sans-serif;color:#17223b">PathSnap · <span style="color:#2196f3">GPS</span> · <span style="color:#20b486">prediction</span>'+reference_legend+'</div>'))
    return m
