import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from fastapi.testclient import TestClient
from pathsnap.data import Trip, Network, read_trip
from pathsnap.decoder import Decoder
from pathsnap.api import app
from pathsnap.maps import make_map

def toy():
    nodes=np.array([[8.,49.],[8.001,49.],[8.002,49.],[8.,49.01],[8.001,49.01]])
    arcs=np.array([[0,1],[1,2],[1,0],[3,4]])
    track=np.array([[8.0002,49.,1.],[8.0008,49.,2.],[8.0015,49.,3.]])
    return Trip('00000000',track,nodes,arcs)

class PipelineTests(unittest.TestCase):
    def test_exact_candidates_match_bruteforce(self):
        trip=toy(); net=Network(trip); ids,d,_=net.candidates(trip.track,k=3)
        for i,p in enumerate(net.project(trip.track[:,:2])):
            exact,_=net.distances(p,np.arange(len(trip.arcs)))
            expected=np.lexsort((np.arange(len(exact)),exact))[:3]
            np.testing.assert_array_equal(ids[i],expected)
            np.testing.assert_allclose(d[i],exact[expected])

    def test_directed_decoder_prefers_reachable_path(self):
        trip=toy();net=Network(trip)
        ids=np.array([[0,3],[0,3],[1,3]])
        frac=np.array([[.2,.2],[.8,.8],[.5,.5]])
        predicted,resets=Decoder(net,max_speed=100).decode(trip.track,ids,frac,np.array([[.99,.01],[.99,.01],[.99,.01]]))
        np.testing.assert_array_equal(predicted,[0,0,1]);self.assertEqual(resets,[])
        transition=Decoder(net).transition(np.array([0]),np.array([3]),np.array([.2]),np.array([.2]),1.,0.)
        self.assertFalse(np.isfinite(transition).any())

    def test_disconnected_candidates_report_reset(self):
        trip=toy();net=Network(trip)
        ids=np.array([[0],[3],[0]])
        _,resets=Decoder(net).decode(trip.track,ids,np.full((3,1),.5),np.ones((3,1)))
        self.assertEqual(resets,[1,2])

    def test_inference_loader_does_not_open_reference(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);trip=toy()
            for ext,a in [('track',trip.track),('nodes',trip.nodes),('arcs',trip.arcs)]: np.savetxt(p/('x.'+ext),a,fmt='%d' if ext=='arcs' else '%.8f')
            (p/'x.route').write_text('deliberately invalid labels')
            self.assertEqual(len(read_trip(p).track),3)
            trip.track[1,2]=trip.track[0,2]; np.savetxt(p/'x.track',trip.track)
            with self.assertRaises(ValueError): read_trip(p)

    def test_api_partition_and_unknown_trip(self):
        client=TestClient(app)
        with patch.dict(os.environ,{'PATHSNAP_PARTITION':'evaluator_test'}):
            response=client.get('/trips');self.assertEqual(response.status_code,200)
            self.assertEqual(response.json()['partition'],'evaluator_test')
            self.assertFalse(response.json()['ready'])
            self.assertEqual(client.post('/predict/123').status_code,404)

    def test_api_map_has_geometry_without_reference_layer(self):
        trip=toy()
        html=make_map(trip,np.array([0,0,1])).get_root().render()
        self.assertIn('MultiLineString',html)
        self.assertIn('Predicted segments',html)
        self.assertNotIn('Reference route',html)
        self.assertNotIn('reference (development only)',html)

if __name__=='__main__': unittest.main()
