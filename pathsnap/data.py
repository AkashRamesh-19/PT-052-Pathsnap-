from dataclasses import dataclass
from pathlib import Path
import hashlib
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from pyproj import CRS, Transformer

ROOT = Path(__file__).resolve().parents[1]

@dataclass
class Trip:
    id: str
    track: np.ndarray
    nodes: np.ndarray
    arcs: np.ndarray

def read_trip(folder):
    """Inference loader deliberately never opens .route files."""
    folder = Path(folder)
    def load(ext, columns, dtype=float):
        files = list(folder.glob('*.' + ext))
        if len(files) != 1:
            raise ValueError(f'{folder}: expected exactly one .{ext} file')
        a = np.loadtxt(files[0], dtype=float, ndmin=2)
        if not len(a) or a.shape[1] != columns or not np.isfinite(a).all():
            raise ValueError(f'{files[0]}: invalid shape or nonfinite values')
        if dtype is int:
            if not np.equal(a, np.floor(a)).all(): raise ValueError('Noninteger road indices')
            a = a.astype(np.int64)
        return a
    track, nodes, arcs = load('track', 3), load('nodes', 2), load('arcs', 2, int)
    for a in (track[:, :2], nodes):
        if (np.abs(a[:, 0]) > 180).any() or (np.abs(a[:, 1]) > 90).any():
            raise ValueError('Invalid longitude/latitude')
    if len(track) < 2 or (np.diff(track[:, 2]) <= 0).any():
        raise ValueError('Need two or more strictly time-ordered observations')
    if not len(arcs) or arcs.min() < 0 or arcs.max() >= len(nodes):
        raise ValueError('Invalid network indices')
    return Trip(folder.name, track, nodes, arcs)

def read_reference(folder, arc_count):
    files = list(Path(folder).glob('*.route'))
    if len(files) != 1:
        raise ValueError(f'{folder}: expected one reference route')
    raw = np.loadtxt(files[0], dtype=float, ndmin=2)
    if not len(raw) or raw.shape[1] != 1 or not np.isfinite(raw).all() or not np.equal(raw,np.floor(raw)).all():
        raise ValueError('Reference must contain one integer arc ID per row')
    route = raw[:,0].astype(np.int64)
    if not len(route) or route.min() < 0 or route.max() >= arc_count:
        raise ValueError('Invalid reference arc indices')
    return route

class Network:
    def __init__(self, trip):
        self.nodes, self.arcs = trip.nodes, trip.arcs
        self.origin = self.nodes.mean(axis=0)
        self.scale = np.array([111320 * np.cos(np.deg2rad(self.origin[1])), 111320.])
        crs = CRS.from_proj4(f'+proj=aeqd +lat_0={self.origin[1]} +lon_0={self.origin[0]} +datum=WGS84 +units=m')
        self.transformer = Transformer.from_crs('EPSG:4326',crs,always_xy=True)
        xy = self.project(self.nodes)
        self.a, self.b = xy[self.arcs[:, 0]], xy[self.arcs[:, 1]]
        self.vec = self.b - self.a
        self.length = np.linalg.norm(self.vec, axis=1)
        self.mid = (self.a + self.b) / 2
        self.tree = cKDTree(self.mid)
        self.max_half = self.length.max() / 2
        self.adj = [[] for _ in self.nodes]
        for (u, v), w in zip(self.arcs, self.length):
            self.adj[u].append((int(v), float(w)))

    def project(self, lonlat):
        a=np.asarray(lonlat)
        x,y=self.transformer.transform(a[...,0],a[...,1])
        return np.stack([x,y],axis=-1)

    def distances(self, p, ids):
        v, a = self.vec[ids], self.a[ids]
        frac = np.clip(np.sum((p - a) * v, axis=-1) / np.maximum(self.length[ids] ** 2, 1e-12), 0, 1)
        d = np.linalg.norm(p - (a + frac[..., None] * v), axis=-1)
        return d, frac

    def candidates(self, track, k=6, subset=None):
        """Exact segment nearest neighbours using midpoint bounding spheres."""
        if k < 1: raise ValueError('Candidate count must be positive')
        points = self.project(track[:, :2])
        allowed = np.arange(len(self.arcs)) if subset is None else np.unique(subset)
        k = min(k, len(allowed))
        tree = self.tree if subset is None else cKDTree(self.mid[allowed])
        half = self.max_half if subset is None else self.length[allowed].max() / 2
        _, seed = tree.query(points, k=k)
        seed = np.asarray(seed).reshape(len(points), k)
        result, distances, fractions = [], [], []
        for p, initial in zip(points, seed):
            initial = allowed[initial]
            radius = self.distances(p, initial)[0].max() + half + 1e-7
            ids = allowed[np.asarray(tree.query_ball_point(p, radius), dtype=int)]
            d, f = self.distances(p, ids)
            order = np.lexsort((ids, d))[:k]
            result.append(ids[order]); distances.append(d[order]); fractions.append(f[order])
        return np.array(result), np.array(distances), np.array(fractions)

def network_key(trip):
    return hashlib.sha256(trip.nodes.tobytes() + trip.arcs.tobytes()).hexdigest()

def inventory(partition):
    rows = []
    for folder in sorted(Path(partition).iterdir()):
        if not folder.is_dir():
            continue
        try:
            t = read_trip(folder); n = Network(t)
            graph = csr_matrix((np.ones(len(t.arcs)), (t.arcs[:, 0], t.arcs[:, 1])), shape=(len(t.nodes),)*2)
            components, _ = connected_components(graph, directed=True, connection='weak')
            degree = np.bincount(t.arcs.ravel(), minlength=len(t.nodes))
            dt = np.diff(t.track[:, 2])
            rows.append(dict(partition=Path(partition).name, trip=t.id, observations=len(t.track), nodes=len(t.nodes), arcs=len(t.arcs), weak_components=components, isolated_nodes=int((degree==0).sum()), max_gap=float(dt.max()), median_gap=float(np.median(dt)), irregular_steps=int((dt!=1).sum()), bbox=[*t.track[:, :2].min(0), *t.track[:, :2].max(0)], reference_available=bool(list(folder.glob('*.route'))), missing_values=0, files={p.name:p.stat().st_size for p in folder.iterdir() if p.is_file()}))
        except (ValueError, OSError) as exc:
            rows.append(dict(partition=Path(partition).name, trip=folder.name, error=str(exc)))
    return rows
