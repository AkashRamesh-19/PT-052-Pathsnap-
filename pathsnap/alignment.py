"""Ordered reference-route alignment used ONLY by training/evaluation, not inference."""
from dataclasses import dataclass
import numpy as np

ALIGNMENT_VERSION = 1

@dataclass
class Alignment:
    labels: np.ndarray
    occurrences: np.ndarray
    confident: np.ndarray
    margin: np.ndarray
    resets: list
    disconnected_reference_steps: int

def align_reference(trip, net, route):
    """Viterbi over route OCCURRENCES, preserving repeated visits and direction.

    Fixed engineering scales (25m observation noise, 50m transition discrepancy,
    80m/s movement cap) are not fitted on held-out data. Confidence is an internal
    alignment margin, not a calibrated probability or verified physical truth.
    """
    route=np.asarray(route,dtype=int)
    near,_,_=net.candidates(trip.track,k=min(10,len(np.unique(route))),subset=route)
    positions={int(a):np.flatnonzero(route==a) for a in np.unique(route)}
    states=[np.sort(np.concatenate([positions[int(a)] for a in row])) for row in near]
    lengths=net.length[route]; starts=np.r_[0.,np.cumsum(lengths)[:-1]]
    broken=trip.arcs[route[:-1],1]!=trip.arcs[route[1:],0]
    component=np.r_[0,np.cumsum(broken)]
    xy=net.project(trip.track[:,:2]); movement=np.linalg.norm(np.diff(xy,axis=0),axis=1)
    distances=[]; progress=[]; emission=[]
    for t,ss in enumerate(states):
        d,f=net.distances(xy[t],route[ss])
        distances.append(d); progress.append(starts[ss]+f*lengths[ss])
        emission.append(np.minimum((d/25.)**2,400.))
    transitions=[]; forward=[emission[0]]; backs=[]; resets=[]
    for t in range(1,len(states)):
        dt=trip.track[t,2]-trip.track[t-1,2]
        delta=progress[t][None,:]-progress[t-1][:,None]
        valid=(states[t][None,:]>=states[t-1][:,None]) & (delta>=-25.) & (delta<=80.*dt+100.)
        valid &= component[states[t]][None,:]==component[states[t-1]][:,None]
        cost=np.where(valid,np.abs(delta-movement[t-1])/50.,np.inf)
        if dt>60 or not np.isfinite(forward[-1][:,None]+cost).any():
            cost=np.zeros_like(cost); resets.append(t)
        total=forward[-1][:,None]+cost
        back=total.argmin(0)
        forward.append(total[back,np.arange(len(back))]+emission[t])
        backs.append(back); transitions.append(cost)
    chosen=[int(forward[-1].argmin())]
    for back in reversed(backs): chosen.append(int(back[chosen[-1]]))
    chosen=np.array(chosen[::-1])
    backward=[None]*len(states); backward[-1]=np.zeros(len(states[-1]))
    for t in range(len(states)-2,-1,-1):
        backward[t]=np.min(transitions[t]+emission[t+1][None,:]+backward[t+1][None,:],axis=1)
    labels=np.array([route[ss[j]] for ss,j in zip(states,chosen)])
    occurrences=np.array([ss[j] for ss,j in zip(states,chosen)])
    margin=np.zeros(len(states)); confident=np.zeros(len(states),dtype=bool)
    for t,ss in enumerate(states):
        scores=forward[t]+backward[t]
        alternative=scores[route[ss]!=labels[t]]
        margin[t]=max(0.,float(alternative.min()-scores[chosen[t]])) if len(alternative) else 1e6
        if not np.isfinite(margin[t]):margin[t]=1e6
        confident[t]=distances[t][chosen[t]]<=60 and margin[t]>=.15
    for t in resets:
        confident[max(0,t-1):min(len(confident),t+2)]=False
    return Alignment(labels,occurrences,confident,margin,resets,int(broken.sum()))
