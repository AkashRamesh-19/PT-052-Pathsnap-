import unittest
import numpy as np
from pathsnap.train import families,metrics

class TrainingPolicyTests(unittest.TestCase):
    def test_histogram_boosting_cannot_randomly_split_candidate_rows(self):
        self.assertIs(families()['HistGradientBoosting'].early_stopping,False)

    def test_candidate_misses_remain_in_accuracy_denominator(self):
        record=dict(ids=np.array([[0,1,2,3,4,5],[0,1,2,3,4,5]]),labels=np.array([0,5]),confident=np.array([True,False]))
        raw=np.array([[.9,.02,.02,.02,.02,.02],[.9,.02,.02,.02,.02,.99]])
        result=metrics(record,raw,k=4)
        self.assertEqual(result['candidate_miss_rate'],.5)
        self.assertEqual(result['top1_accuracy'],.5)
        self.assertEqual(result['confident_coverage'],.5)
        self.assertEqual(result['confident_top1_accuracy'],1.)

    def test_primary_accuracy_does_not_reward_predicting_only_negative_candidates(self):
        record=dict(ids=np.tile(np.arange(12),(2,1)),labels=np.array([0,0]),confident=np.array([True,True]))
        raw=np.ones((2,12))*.01;raw[:,1]=.02
        result=metrics(record,raw,k=12)
        self.assertEqual(result['top1_accuracy'],0.)
        self.assertEqual(result['candidate_miss_rate'],0.)

if __name__=='__main__':unittest.main()
