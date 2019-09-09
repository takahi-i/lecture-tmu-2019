import unittest

from lecture_tmu_2019.ml import ReputationClassifier


class TestSample(unittest.TestCase):
    def test_fit(self):
        classiier = ReputationClassifier()
        best_score = classiier.fit()
        self.assertGreaterEqual(best_score, 0.7)
