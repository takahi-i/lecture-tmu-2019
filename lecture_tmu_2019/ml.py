import glob
import os
import sys

import numpy as np
from sklearn import svm, grid_search
from sklearn.feature_extraction import DictVectorizer

from lecture_tmu_2019.utils import text_reader, word_counter, get_unigram, DATA_NUM

SEARCH_PARAMETERS = [{'kernel': ['rbf'], 'gamma': [1e-2, 1e-3, 1e-4], 'C': [0.1, 1, 10, 100, 1000]},
        {'kernel': ['linear'], 'C': [0.1, 1, 10, 100, 1000]}]

DATASET_BASE_PATH = "dataset"
POSITIVE_FILES = "{}/tokens/pos/*".format(DATASET_BASE_PATH)
NEGATIVE_FILES = "{}/tokens/neg/*".format(DATASET_BASE_PATH)


class ReputationClassifier:
    def __init__(self):
        self.feature_vectors = self._generate_feature_vectors(self._load_data())

    def _load_data(self):
        print(os.listdir(os.path.normpath(DATASET_BASE_PATH)))
        neg_files = glob.glob(os.path.normpath(NEGATIVE_FILES))
        pos_files = glob.glob(os.path.normpath(POSITIVE_FILES))
        text_reader(neg_files[11])
        word_counter("I am YK. I love data analysis using python.")
        unigrams = get_unigram(neg_files[:DATA_NUM]) + get_unigram(pos_files[:DATA_NUM])
        print("data size :", sys.getsizeof(unigrams) / 1000000, "[MB]")
        return unigrams

    def _generate_feature_vectors(self, unigrams):
        vec = DictVectorizer()
        return vec.fit_transform(unigrams)

    def fit(self, search_parameters=SEARCH_PARAMETERS):
        labels = np.r_[np.tile(0, DATA_NUM), np.tile(1, DATA_NUM)]
        model = svm.SVC()
        clf = grid_search.GridSearchCV(model, search_parameters)
        clf.fit(self.feature_vectors, labels)
        print("best parameters : ", clf.best_params_)
        print("best scores : ", clf.best_score_)
