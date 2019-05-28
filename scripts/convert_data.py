import os

print(os.listdir(os.path.normpath("./")))
print(os.listdir(os.path.normpath("dataset/")))


def set_locale():
    default = os.environ.get('LC_ALL')
    print("Your default locale is", default)
    if default is None:
        os.environ.setdefault('LC_ALL', 'ja_JP.UTF-8')
        print("Your locale is set as ja_JP.UTF-8")


set_locale()

import glob

neg_files = glob.glob(os.path.normpath("dataset/tokens/neg/*"))
pos_files = glob.glob(os.path.normpath("dataset/tokens/pos/*"))

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

print(word_counter("I am YK. I love data analysis using python."))

