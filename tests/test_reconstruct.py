import unittest
import numpy as np
from pathsnap.data import Trip,Network
from pathsnap.reconstruct import reconstruct_sections

class ReconstructionTests(unittest.TestCase):
    def test_missing_intermediate_arc_is_reconstructed(self):
        nodes=np.array([[8.,49.],[8.001,49.],[8.002,49.],[8.003,49.]])
        arcs=np.array([[0,1],[1,2],[2,3]])
        track=np.array([[8.0005,49.,1.],[8.0025,49.,4.]])
        trip=Trip('test',track,nodes,arcs);net=Network(trip)
        self.assertEqual(reconstruct_sections(trip,net,[0,2],[]),[[0,1,2]])
        self.assertEqual(reconstruct_sections(trip,net,[0,2],[1]),[[0],[2]])

    def test_disconnected_segments_never_get_an_invented_connector(self):
        nodes=np.array([[8.,49.],[8.001,49.],[8.002,49.],[8.003,49.]])
        arcs=np.array([[0,1],[2,3]]);track=np.array([[8.0005,49.,1.],[8.0025,49.,4.]])
        trip=Trip('test',track,nodes,arcs)
        self.assertEqual(reconstruct_sections(trip,Network(trip),[0,1],[]),[[0],[1]])

if __name__=='__main__':unittest.main()
