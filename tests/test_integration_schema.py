import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import joblib
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from fastapi.testclient import TestClient
from pathsnap import api
from pathsnap.features import FEATURE_NAMES,FEATURE_SCHEMA_VERSION
from pathsnap.data import read_trip

class InferenceIsolationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        art=self.root/'artifacts';art.mkdir()
        model=DecisionTreeClassifier(random_state=352).fit(np.array([[0.]*20,[1.]*20]),[0,1])
        digest=hashlib.sha256(json.dumps(dict(names=FEATURE_NAMES,version=FEATURE_SCHEMA_VERSION,projection='WGS84 local azimuthal equidistant; always_xy'),sort_keys=True).encode()).hexdigest()
        model.pathsnap_schema_hash_=digest
        config=dict(model='test',feature_names=FEATURE_NAMES,feature_schema_version=FEATURE_SCHEMA_VERSION,feature_schema_hash=digest,k=6,decoder=dict(beta=60.,max_speed=55.,gap_reset=60.))
        joblib.dump(model,art/'model.joblib');(art/'config.json').write_text(json.dumps(config))
        self.folder=self.root/'evaluator_test/00000000';self.folder.mkdir(parents=True)
        np.savetxt(self.folder/'x.nodes',[[8,49],[8.001,49],[8.002,49]])
        np.savetxt(self.folder/'x.arcs',[[0,1],[1,2]],fmt='%d')
        np.savetxt(self.folder/'x.track',[[8.0002,49,1],[8.0008,49,2],[8.0012,49,3]])
        (self.folder/'x.route').write_text('INVALID HIDDEN REFERENCE: MUST NEVER BE READ')
        self.root_patch=patch.object(api,'ROOT',self.root);self.root_patch.start();api.assets.cache_clear()

    def tearDown(self):
        api.assets.cache_clear();self.root_patch.stop();self.temp.cleanup()

    def test_evaluator_inference_does_not_read_labels_even_if_scoring_requested(self):
        with patch.dict(os.environ,{'PATHSNAP_PARTITION':'evaluator_test','PATHSNAP_SCORE_STUDENT':'1'}),patch.object(api,'read_reference',side_effect=AssertionError('Reference leakage')):
            response=TestClient(api.app).post('/predict/00000000')
            self.assertEqual(response.status_code,200,response.text[:200])
            self.assertEqual(response.json()['metrics'],{})
            self.assertEqual(len(response.json()['predicted_arc_ids']),3)
            self.assertNotIn('Reference route',response.json()['map_html'])

    def test_health_validates_schema_hash_not_just_file_presence(self):
        client=TestClient(api.app);self.assertEqual(client.get('/health').json()['feature_count'],20)
        cfgpath=self.root/'artifacts/config.json';cfg=json.loads(cfgpath.read_text());cfg['feature_names']=cfg['feature_names'][::-1];cfgpath.write_text(json.dumps(cfg));api.assets.cache_clear()
        health=client.get('/health').json();self.assertFalse(health['model_ready'])

    def test_fractional_arc_index_is_not_silently_truncated(self):
        (self.folder/'x.arcs').write_text('0.5 1\n')
        with self.assertRaises(ValueError):read_trip(self.folder)

if __name__=='__main__':unittest.main()
