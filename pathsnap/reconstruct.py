"""Expand predicted states into directed road paths; no reference-route access."""
import heapq
import numpy as np

def reconstruct_sections(trip,net,predicted,resets,max_speed=55.,gap_reset=60.):
    predicted=np.asarray(predicted,dtype=int)
    if len(predicted)!=len(trip.track):raise ValueError('Need one predicted state per observation')
    _,fractions=net.distances(net.project(trip.track[:,:2]),predicted)
    arc_for_pair={}
    for i,(u,v) in enumerate(net.arcs):arc_for_pair.setdefault((int(u),int(v)),i)
    def bridge(source,target,cutoff):
        if source==target:return []
        distances={source:0.};previous={};queue=[(0.,source)]
        while queue:
            d,u=heapq.heappop(queue)
            if d!=distances[u]:continue
            if u==target:
                route=[]
                while u!=source:
                    old=previous[u];route.append(arc_for_pair[(old,u)]);u=old
                return route[::-1]
            for v,w in net.adj[u]:
                nd=d+w
                if nd<=cutoff and nd<distances.get(v,np.inf):
                    distances[v]=nd;previous[v]=u;heapq.heappush(queue,(nd,v))
        return None
    sections=[[int(predicted[0])]];reset_set=set(resets)
    for t in range(1,len(predicted)):
        a,b=int(predicted[t-1]),int(predicted[t]);dt=trip.track[t,2]-trip.track[t-1,2]
        if t in reset_set or dt>gap_reset:
            sections.append([b]);continue
        if a==b and (fractions[t]-fractions[t-1])*net.length[a]>=-15.:continue
        allowance=min(10000,max(80,max_speed*dt+60))-(1-fractions[t-1])*net.length[a]-fractions[t]*net.length[b]
        links=bridge(int(net.arcs[a,1]),int(net.arcs[b,0]),allowance) if allowance>=0 else None
        if links is None:sections.append([b])
        else:sections[-1].extend([*links,b])
    return sections
