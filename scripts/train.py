import os
import glob
import sys

import numpy as np
from sklearn import svm
from sklearn import grid_search
from sklearn.feature_extraction import DictVectorizer

from lecture_tmu_2019.utils import set_locale, text_reader, word_counter, get_unigram, cross_validate, DATA_NUM

print(os.listdir(os.path.normpath("dataset/")))

set_locale()

neg_files = glob.glob(os.path.normpath("dataset/tokens/neg/*"))
pos_files = glob.glob(os.path.normpath("dataset/tokens/pos/*"))

print(neg_files[0:2])
print(pos_files[0:2])

text_reader(neg_files[11])

word_counter("I am YK. I love data analysis using python.")

unigrams_data = get_unigram(neg_files[:DATA_NUM]) + get_unigram(pos_files[:DATA_NUM])

print("data size :", sys.getsizeof(unigrams_data) / 1000000, "[MB]")

vec = DictVectorizer()
feature_vectors_csr = vec.fit_transform(unigrams_data)

feature_vectors = vec.fit_transform(unigrams_data).toarray()
print("data dimension :", feature_vectors.shape)
print(feature_vectors[0])
print("data size :", sys.getsizeof(feature_vectors) / 1000000, "[MB]")

labels = np.r_[np.tile(0, DATA_NUM), np.tile(1, DATA_NUM)]

print(labels[0], labels[DATA_NUM-1], labels[DATA_NUM], labels[2*DATA_NUM-1])

np.random.seed(7789)

shuffle_order = np.random.choice(2*DATA_NUM, 2*DATA_NUM, replace=False)

print("length :", len(shuffle_order))
print("first 10 elements :", shuffle_order[0:10])

one_third_size = int(2*DATA_NUM / 3.)
print("one third of the length :", one_third_size)

print("# of '1' in 1st set :", np.sum(labels[shuffle_order[:one_third_size]]))
print("# of '1' in 2nd set :", np.sum(labels[shuffle_order[one_third_size:2*one_third_size]]))
print("# of '1' in 3rd set :", np.sum(labels[shuffle_order[2*one_third_size:]]))

N_FOLDS = 3

ans, corr = cross_validate(N_FOLDS, feature_vectors_csr, labels, shuffle_order, method='SVM', parameters=None)

print("average precision : ", np.around(100.*sum(corr)/sum(ans), decimals=1), "%")

search_parameters = [
    {'kernel': ['rbf'], 'gamma': [1e-2, 1e-3, 1e-4], 'C': [0.1, 1, 10, 100, 1000]},
    {'kernel': ['linear'], 'C': [0.1, 1, 10, 100, 1000]}
]

model = svm.SVC()
clf = grid_search.GridSearchCV(model, search_parameters)
clf.fit(feature_vectors_csr, labels)

print("best parameters : ", clf.best_params_)
print("best scores : ", clf.best_score_)
