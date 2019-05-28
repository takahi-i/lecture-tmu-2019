import glob
import sys
import os
import collections

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn import svm, naive_bayes
from sklearn.ensemble import RandomForestClassifier
from sklearn import grid_search
import matplotlib  # not used in this notebook
import pandas as pd  # not used in this notebook


def set_locale():
    default = os.environ.get('LC_ALL')
    print("Your default locale is", default)
    if default is None:
        os.environ.setdefault('LC_ALL', 'ja_JP.UTF-8')
        print("Your locale is set as ja_JP.UTF-8")


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


print(os.listdir(os.path.normpath("./")))
print(os.listdir(os.path.normpath("dataset/")))

set_locale()

neg_files = glob.glob(os.path.normpath("dataset/tokens/neg/*"))
pos_files = glob.glob(os.path.normpath("dataset/tokens/pos/*"))

print(neg_files[0:2])
print(pos_files[0:2])

print(word_counter("I am YK. I love data analysis using python."))

DATA_NUM = 700

unigrams_data = get_unigram(neg_files[:DATA_NUM]) + get_unigram(pos_files[:DATA_NUM])

print(unigrams_data[0])
print("data size :", sys.getsizeof(unigrams_data) / 1000000, "[MB]")

vec = DictVectorizer()
feature_vectors_csr = vec.fit_transform(unigrams_data)

print(feature_vectors_csr)

feature_vectors = vec.fit_transform(unigrams_data).toarray()
print("data dimension :", feature_vectors.shape)
print(feature_vectors[0])
print("data size :", sys.getsizeof(feature_vectors) / 1000000, "[MB]")

labels = np.r_[np.tile(0, DATA_NUM), np.tile(1, DATA_NUM)]
print(labels[0], labels[DATA_NUM - 1], labels[DATA_NUM], labels[2 * DATA_NUM - 1])

np.random.seed(7789)
shuffle_order = np.random.choice(2 * DATA_NUM, 2 * DATA_NUM, replace=False)

print("length :", len(shuffle_order))
print("first 10 elements :", shuffle_order[0:10])

one_third_size = int(2 * DATA_NUM / 3.)
print("one third of the length :", one_third_size)

print("# of '1' in 1st set :", np.sum(labels[shuffle_order[:one_third_size]]))
print("# of '1' in 2nd set :", np.sum(labels[shuffle_order[one_third_size:2 * one_third_size]]))
print("# of '1' in 3rd set :", np.sum(labels[shuffle_order[2 * one_third_size:]]))
