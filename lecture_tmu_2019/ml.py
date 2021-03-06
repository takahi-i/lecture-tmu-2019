import glob
import os
import sys
import numpy as np
import hideout
from sklearn import svm, grid_search
from sklearn.feature_extraction import DictVectorizer
from typing import Dict, List

from lecture_tmu_2019.settings import SEARCH_PARAMETERS, DATASET_BASE_PATH, POSITIVE_FILES, NEGATIVE_FILES, DATA_NUM
from lecture_tmu_2019.utils import text_reader, word_counter, get_unigram


class ReputationClassifier:
    def __init__(self):
        self.vec, self.feature_vectors = hideout.resume_or_generate(
            label="vectorizer",
            func=self._vectorizer
        )
        self.clf = None
        self.model = None

    def _vectorizer(self):
        vec = DictVectorizer()
        feature_vector = vec.fit_transform(self._load_data())
        return [vec, feature_vector]

    def _load_data(self):
        print(os.listdir(os.path.normpath(DATASET_BASE_PATH)))
        neg_files = glob.glob(os.path.normpath(NEGATIVE_FILES))
        pos_files = glob.glob(os.path.normpath(POSITIVE_FILES))
        text_reader(neg_files[11])
        word_counter("I am YK. I love data analysis using python.")
        unigrams = get_unigram(neg_files[:DATA_NUM]) + get_unigram(pos_files[:DATA_NUM])
        print("data size :", sys.getsizeof(unigrams) / 1000000, "[MB]")
        return unigrams

    def fit(self, search_parameters: List = SEARCH_PARAMETERS) -> float:
        clf, self.model = hideout.resume_or_generate(
            label="classifier",
            func=self._fit,
            func_args={"search_parameters": search_parameters}
        )
        print("best parameters : ", clf.best_params_)
        print("best scores : ", clf.best_score_)
        return clf.best_score_

    def _fit(self, search_parameters: Dict) -> List:
        labels = np.r_[np.tile(0, DATA_NUM), np.tile(1, DATA_NUM)]
        clf = grid_search.GridSearchCV(svm.SVC(), search_parameters)
        clf.fit(self.feature_vectors, labels)
        return [clf, clf.best_estimator_]

    def predict(self, text: Dict) -> np.ndarray:
        vector = self.vec.transform(text)
        return self.model.predict(vector)
