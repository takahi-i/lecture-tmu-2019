import os
import collections
from typing import List, Dict


def set_locale():
    default = os.environ.get('LC_ALL')
    print( "Your default locale is", default )
    if default is None:
        os.environ.setdefault('LC_ALL', 'ja_JP.UTF-8')
        print( "Your locale is set as ja_JP.UTF-8" )


def text_reader(file_path: str):
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            print(line)


def word_counter(string: str) -> Dict:
    words = string.strip().split()
    count_dict = collections.Counter(words)
    return dict(count_dict)


def get_unigram(file_path: str) -> List:
    result = []
    for file in file_path:
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                count_dict = word_counter(line)
                result.append(count_dict)
    return result


def get_unigram_from_text(text: str) -> List:
    result = []
    for line in text:
        count_dict = word_counter(line)
        result.append(count_dict)
    return result

