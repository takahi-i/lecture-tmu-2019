import unittest

from lecture_tmu_2019.ml import ReputationClassifier


class TestSample(unittest.TestCase):
    def test_fit(self):
        classiier = ReputationClassifier()
        best_score = classiier.fit()
        self.assertGreaterEqual(0.7, best_score)  # NOTE: FAIL!
