import os
import sys
import collections

import numpy as np
from sklearn import svm, naive_bayes
from sklearn.ensemble import RandomForestClassifier

from lecture_tmu_2019.settings import DATA_NUM


def set_locale():
    default = os.environ.get('LC_ALL')
    print( "Your default locale is", default )
    if default is None:
        os.environ.setdefault('LC_ALL', 'ja_JP.UTF-8')
        print( "Your locale is set as ja_JP.UTF-8" )


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


def N_splitter(seq, N):
    avg = len(seq) / float(N)
    out = []
    last = 0.0

    while last < len(seq):
        out.append( seq[int(last):int(last + avg)] )
        last += avg

    return np.array(out)


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

    n_splits = N_splitter(range(2 * DATA_NUM), n_folds)

    for i in range(n_folds):
        print( "Executing {0}th set...".format(i+1) )

        test_elems = shuffle_order[ n_splits[i] ]
        train_elems = np.array([])
        train_set = n_splits[ np.arange(n_folds) !=i ]
        for j in train_set:
            train_elems = np.r_[ train_elems, shuffle_order[j] ]
        train_elems = train_elems.astype(np.integer)

        # train
        model = train_model(feature_vectors[train_elems], labels[train_elems], method, parameters)
        # predict
        predictions = predict(model, feature_vectors[test_elems])
        # evaluate
        test_num, correct_num = evaluate_model(predictions, labels[test_elems])
        result_test_num.append( test_num )
        result_correct_num.append( correct_num )

    return result_test_num, result_correct_num
