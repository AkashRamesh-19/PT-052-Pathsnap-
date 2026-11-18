import tempfile
import unittest
from pathlib import Path
import numpy as np
from pathsnap.data import Trip, Network, read_trip
from pathsnap.features import prepare, FEATURE_NAMES, LEGACY_FEATURE_NAMES


def fixture():
    nodes = np.array([[8., 49.], [8.001, 49.], [8.002, 49.], [8., 49.01], [8.001, 49.01]])
    arcs = np.array([[0, 1], [1, 2], [1, 0], [3, 4]])
    track = np.array([[8.0002, 49., 1.], [8.0008, 49., 2.], [8.0015, 49., 3.]])
    return Trip('00000000', track, nodes, arcs)


class DataFeatureTests(unittest.TestCase):
    def test_candidates_agree_with_brute_force(self):
        trip = fixture(); network = Network(trip)
        ids, distances, _ = network.candidates(trip.track, k=3)
        for i, point in enumerate(network.project(trip.track[:, :2])):
            exact, _ = network.distances(point, np.arange(len(trip.arcs)))
            expected = np.lexsort((np.arange(len(exact)), exact))[:3]
            np.testing.assert_array_equal(ids[i], expected)
            np.testing.assert_allclose(distances[i], exact[expected])

    def test_loader_ignores_labels_and_rejects_bad_timestamps(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp); trip = fixture()
            for ext, array in [('track', trip.track), ('nodes', trip.nodes), ('arcs', trip.arcs)]:
                np.savetxt(folder / ('trip.' + ext), array, fmt='%d' if ext == 'arcs' else '%.8f')
            (folder / 'trip.route').write_text('Invalid reference deliberately ignored')
            self.assertEqual(len(read_trip(folder).track), 3)
            trip.track[1, 2] = trip.track[0, 2]
            np.savetxt(folder / 'trip.track', trip.track)
            with self.assertRaises(ValueError):
                read_trip(folder)

    def test_features_preserve_projection_and_motion(self):
        trip = fixture(); network = Network(trip)
        ids, fractions, features = prepare(trip, network, k=3)
        self.assertEqual(features.shape, (3, 3, 20))
        self.assertTrue(np.isfinite(features).all())
        self.assertTrue(((fractions >= 0) & (fractions <= 1)).all())
        # Each observation lies on a forward segment in the fixture.
        for t in range(3):
            forward = 0 if t < 2 else 1
            j = np.flatnonzero(ids[t] == forward)[0]
            # AEQD projects an ellipsoidal constant-latitude curve to a line approximation.
            self.assertLess(features[t, j, 0], .001)
            self.assertAlmostEqual(features[t, j, 2], 1., places=6)

    def test_motion_context_and_endpoint_geometry(self):
        trip = fixture(); network = Network(trip)
        ids, fractions, x = prepare(trip, network, k=3)
        index = {name: i for i, name in enumerate(FEATURE_NAMES)}
        self.assertEqual(len(set(FEATURE_NAMES)), 20)
        speed = x[:, 0, index['speed_mps']]
        self.assertAlmostEqual(x[2, 0, index['acceleration_mps2']], speed[2]-speed[1])
        self.assertAlmostEqual(x[2, 0, index['speed_mean_5_mps']], np.mean(speed))
        self.assertAlmostEqual(x[2, 0, index['speed_std_5_mps']], np.std(speed))
        np.testing.assert_allclose(x[:,:,index['distance_from_start_m']]+x[:,:,index['distance_to_end_m']],network.length[ids])
        for t in range(3):
            for j, arc in enumerate(ids[t]):
                u, v = trip.arcs[arc]
                self.assertEqual(x[t,j,index['start_in_degree']], (trip.arcs[:,1]==u).sum())
                self.assertEqual(x[t,j,index['end_out_degree']], (trip.arcs[:,0]==v).sum())

    def test_stationary_zero_length_and_gap_reset(self):
        trip = fixture()
        trip.track[:, :2] = trip.track[0, :2]
        trip.track[2, 2] = 100
        trip.arcs = np.vstack([trip.arcs, [0, 0]])
        _, _, x = prepare(trip, Network(trip), k=1)
        self.assertTrue(np.isfinite(x).all())
        for name in ['speed_mps','acceleration_mps2','turn_angle_rad','speed_std_5_mps','candidate_distance_gap_m']:
            np.testing.assert_allclose(x[:,:,FEATURE_NAMES.index(name)],0)

    def test_legacy_schema_selection_and_unknown_names(self):
        trip = fixture(); network = Network(trip)
        _, _, full = prepare(trip, network, k=3)
        _, _, legacy = prepare(trip, network, k=3, feature_names=LEGACY_FEATURE_NAMES)
        np.testing.assert_allclose(legacy, full[:,:,:8])
        with self.assertRaises(ValueError):
            prepare(trip, network, k=3, feature_names=['not_a_feature'])


if __name__ == '__main__':
    unittest.main()
