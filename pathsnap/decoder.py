from functools import lru_cache
import heapq
import numpy as np

class Decoder:
    def __init__(self, network, beta=60., max_speed=55., gap_reset=60., emission_weight=1., transition_weight=1.):
        self.net, self.beta, self.max_speed, self.gap_reset = network, beta, max_speed, gap_reset
        self.emission_weight, self.transition_weight = emission_weight, transition_weight
        if beta<=0 or max_speed<=0 or gap_reset<=0 or emission_weight<=0 or transition_weight<0:
            raise ValueError('Invalid decoder parameters')

    @lru_cache(maxsize=512)
    def reachable(self, source, cutoff):
        distances = {source: 0.}
        queue = [(0., source)]
        while queue:
            d, u = heapq.heappop(queue)
            if d != distances[u]:
                continue
            for v, w in self.net.adj[u]:
                nd = d+w
                if nd <= cutoff and nd < distances.get(v, np.inf):
                    distances[v] = nd
                    heapq.heappush(queue, (nd, v))
        return distances

    def road_distances(self, prev, current, pf, cf, cutoff):
        n = self.net
        roads = np.full((len(prev), len(current)), np.inf)
        for i, a in enumerate(prev):
            reach = self.reachable(int(n.arcs[a, 1]), cutoff)
            for j, b in enumerate(current):
                delta=(cf[j]-pf[i])*n.length[a]
                if a == b and delta >= -15.:
                    # Local projected GPS jitter is not a full road-network loop.
                    road = abs(delta)
                else:
                    road = (1-pf[i])*n.length[a] + reach.get(int(n.arcs[b, 0]), np.inf) + cf[j]*n.length[b]
                if road <= cutoff:
                    roads[i,j]=road
        return roads

    def transition(self, prev, current, pf, cf, dt, displacement):
        cutoff=float(min(10000,max(80,self.max_speed*dt+60)))
        roads=self.road_distances(prev,current,pf,cf,cutoff)
        return -np.abs(roads-displacement)/(self.beta*np.sqrt(max(dt,1.)))

    def lattice(self, track, ids, fractions):
        xy=self.net.project(track[:,:2]); roads=[]
        dt=np.diff(track[:,2]); displacement=np.linalg.norm(np.diff(xy,axis=0),axis=1)
        for t,gap in enumerate(dt,1):
            if gap>90:
                roads.append(None)
            else:
                cutoff=float(min(10000,max(80,90*gap+60)))
                roads.append(self.road_distances(ids[t-1],ids[t],fractions[t-1],fractions[t],cutoff).astype(np.float32))
        return dict(roads=roads,dt=dt,displacement=displacement)

    def decode_lattice(self, ids, p, lattice):
        logp=np.log(np.maximum(p,1e-12))*self.emission_weight
        score=logp[0].copy(); backs=[]; resets=[]
        for t,(roads,dt,displacement) in enumerate(zip(lattice['roads'],lattice['dt'],lattice['displacement']),1):
            if dt>self.gap_reset or roads is None:
                transition=np.zeros((ids.shape[1],)*2);resets.append(t)
            else:
                valid=np.isfinite(roads)&(roads<=min(10000,max(80,self.max_speed*dt+60)))
                discrepancy=np.abs(np.where(valid,roads,0.)-displacement)
                transition=np.where(valid,-self.transition_weight*discrepancy/(self.beta*np.sqrt(max(dt,1.))),-np.inf)
            total=score[:,None]+transition
            if not np.isfinite(total).any():
                total=np.broadcast_to(score[:,None],transition.shape).copy()
                if t not in resets:resets.append(t)
            back=np.argmax(total,axis=0)
            score=total[back,np.arange(len(back))]+logp[t]
            score-=np.max(score)
            backs.append(back)
        chosen=[int(np.argmax(score))]
        for back in reversed(backs):chosen.append(int(back[chosen[-1]]))
        return ids[np.arange(len(ids)),chosen[::-1]],resets

    def decode(self, track, ids, fractions, p):
        return self.decode_lattice(ids,p,self.lattice(track,ids,fractions))

    def continuity(self, track, ids, fractions, chosen):
        # Reachability at the observed time interval, including interior-arc offsets.
        count = 0
        for t in range(1, len(chosen)):
            a = int(np.flatnonzero(ids[t-1] == chosen[t-1])[0])
            b = int(np.flatnonzero(ids[t] == chosen[t])[0])
            tr = self.transition(ids[t-1, a:a+1], ids[t, b:b+1], fractions[t-1, a:a+1], fractions[t, b:b+1], track[t, 2]-track[t-1, 2], 0)
            count += bool(np.isfinite(tr[0, 0]))
        return count / max(1, len(chosen)-1)
