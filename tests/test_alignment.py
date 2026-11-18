import tempfile
import unittest
from pathlib import Path
import numpy as np
from pathsnap.data import Trip,Network,read_reference,read_trip
from pathsnap.alignment import align_reference

class AlignmentTests(unittest.TestCase):
    def test_repeated_arcs_keep_occurrence_order(self):
        nodes=np.array([[8.,49.],[8.001,49.],[8.001,49.001],[8.,49.001]])
        arcs=np.array([[0,1],[1,2],[2,3],[3,0]])
        route=np.array([0,1,2,3,0,1])
        points=nodes[arcs[route]].mean(axis=1)
        track=np.column_stack([points,np.arange(6)*5.])
        trip=Trip('test',track,nodes,arcs);a=align_reference(trip,Network(trip),route)
        np.testing.assert_array_equal(a.labels,route)
        np.testing.assert_array_equal(a.occurrences,np.arange(6))
        self.assertEqual(a.resets,[])

    def test_fractional_and_multicolumn_reference_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'x.route'
            for text in ['0.5\n','0 1\n','nan\n','-1\n','5\n']:
                p.write_text(text)
                with self.assertRaises(ValueError):read_reference(tmp,4)

    def test_reference_direction_uses_sequence(self):
        nodes=np.array([[8.,49.],[8.001,49.]])
        arcs=np.array([[0,1],[1,0]])
        track=np.array([[8.0002,49.,0.],[8.0008,49.,2.],[8.0008,49.,4.],[8.0002,49.,6.]])
        trip=Trip('test',track,nodes,arcs);a=align_reference(trip,Network(trip),np.array([0,1]))
        self.assertEqual(a.labels[0],0);self.assertEqual(a.labels[-1],1)
        self.assertTrue(np.isfinite(a.margin).all())

if __name__=='__main__':unittest.main()
