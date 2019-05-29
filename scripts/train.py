import os

print( os.listdir(os.path.normpath("dataset/")) )

def set_locale():
    default = os.environ.get('LC_ALL')
    print( "Your default locale is", default )
    if default is None:
        os.environ.setdefault('LC_ALL', 'ja_JP.UTF-8')
        print( "Your locale is set as ja_JP.UTF-8" )

set_locale()

import glob

neg_files = glob.glob( os.path.normpath("dataset/tokens/neg/*"))
pos_files = glob.glob( os.path.normpath("dataset/tokens/pos/*"))

print(neg_files[0:2])
print(pos_files[0:2])

import sys

def text_reader(file_path):
    python_version = sys.version_info.major

    if python_version >= 3:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                print(line)
    else:
        with open(file_path, 'r') as f:
            for line in f:
                print(line)


text_reader(neg_files[11])

import matplotlib # not used in this notebook
import pandas as pd # not used in this notebook

import collections
import numpy as np

from sklearn.feature_extraction import DictVectorizer

from sklearn import svm, naive_bayes
from sklearn.ensemble import RandomForestClassifier

from sklearn import grid_search

def word_counter(string):
    words = string.strip().split()
    count_dict = collections.Counter(words)
    return dict(count_dict)

def get_unigram(file_path):
    result = []
    python_version = sys.version_info.major

    if python_version >= 3:
        for file in file_path:
            with open(file, 'r', encoding='utf-8') as f:
                for line in f:
                    count_dict = word_counter(line)
                    result.append(count_dict)
    else:
        for file in file_path:
            with open(file, 'r') as f:
                for line in f:
                    count_dict = word_counter(line)
                    result.append(count_dict)

    return result


word_counter("I am YK. I love data analysis using python.")

DATA_NUM = 700

unigrams_data = get_unigram(neg_files[:DATA_NUM]) + get_unigram(pos_files[:DATA_NUM])

print( unigrams_data[0] )
print( "data size :", sys.getsizeof(unigrams_data) / 1000000, "[MB]" )

vec = DictVectorizer()
feature_vectors_csr = vec.fit_transform( unigrams_data )

feature_vectors_csr

feature_vectors = vec.fit_transform( unigrams_data ).toarray()
print( "data dimension :", feature_vectors.shape )
print( feature_vectors[0] )
print( "data size :", sys.getsizeof(feature_vectors) / 1000000, "[MB]" )

labels = np.r_[np.tile(0, DATA_NUM), np.tile(1, DATA_NUM)]

print( labels[0], labels[DATA_NUM-1], labels[DATA_NUM], labels[2*DATA_NUM-1]  )

np.random.seed(7789)

shuffle_order = np.random.choice( 2*DATA_NUM, 2*DATA_NUM, replace=False )

print( "length :", len(shuffle_order) )
print( "first 10 elements :", shuffle_order[0:10] )

one_third_size = int( 2*DATA_NUM / 3. )
print( "one third of the length :", one_third_size )

print( "# of '1' in 1st set :", np.sum( labels[ shuffle_order[:one_third_size] ]  ) )
print( "# of '1' in 2nd set :", np.sum( labels[ shuffle_order[one_third_size:2*one_third_size] ]  ) )
print( "# of '1' in 3rd set :", np.sum( labels[ shuffle_order[2*one_third_size:] ]  ) )

def N_splitter(seq, N):
    avg = len(seq) / float(N)
    out = []
    last = 0.0

    while last < len(seq):
        out.append( seq[int(last):int(last + avg)] )
        last += avg

    return np.array(out)

N_splitter(range(14), 3)


def train_model(features, labels, method='SVM', parameters=None):
    ### set the model
    if method == 'SVM':
        model = svm.SVC()
    elif method == 'NB':
        model = naive_bayes.GaussianNB()
    elif method == 'RF':
        model = RandomForestClassifier()
    else:
        print("Set method as SVM (for Support vector machine), NB (for Naive Bayes) or RF (Random Forest)")
    ### set parameters if exists
    if parameters:
        model.set_params(**parameters)
    ### train the model
    model.fit( features, labels )
    ### return the trained model
    return model

def predict(model, features):
    predictions = model.predict( features )
    return predictions

def evaluate_model(predictions, labels):
    data_num = len(labels)
    correct_num = np.sum( predictions == labels )
    return data_num, correct_num

def cross_validate(n_folds, feature_vectors, labels, shuffle_order, method='SVM', parameters=None):
    result_test_num = []
    result_correct_num = []

    n_splits = N_splitter( range(2*DATA_NUM), n_folds )

    for i in range(n_folds):
        print( "Executing {0}th set...".format(i+1) )

        test_elems = shuffle_order[ n_splits[i] ]
        train_elems = np.array([])
        train_set = n_splits[ np.arange(n_folds) !=i ]
        for j in train_set:
            train_elems = np.r_[ train_elems, shuffle_order[j] ]
        train_elems = train_elems.astype(np.integer)

        # train
        model = train_model( feature_vectors[train_elems], labels[train_elems], method, parameters )
        # predict
        predictions = predict( model, feature_vectors[test_elems] )
        # evaluate
        test_num, correct_num = evaluate_model( predictions, labels[test_elems] )
        result_test_num.append( test_num )
        result_correct_num.append( correct_num )

    return result_test_num, result_correct_num

N_FOLDS = 3

ans,corr = cross_validate(N_FOLDS, feature_vectors_csr, labels, shuffle_order, method='SVM', parameters=None)

print( "average precision : ", np.around( 100.*sum(corr)/sum(ans), decimals=1 ), "%" )

search_parameters = [
    {'kernel': ['rbf'], 'gamma': [1e-2, 1e-3, 1e-4], 'C': [0.1, 1, 10, 100, 1000]},
    {'kernel': ['linear'], 'C': [0.1, 1, 10, 100, 1000]}
]

model = svm.SVC()
clf = grid_search.GridSearchCV(model, search_parameters)
clf.fit( feature_vectors_csr, labels )

print("best paremters : ", clf.best_params_)
print("best scores : ", clf.best_score_)
