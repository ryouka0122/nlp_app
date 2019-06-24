# -*- coding: utf-8 -*-
import numpy as np
import csv
import lang.mecab as mecab


def load_data(filename: str, encoding='utf-8') -> tuple:
    data_list = []
    answer_list = []
    with open(filename, 'r', encoding=encoding) as f:
        reader = csv.reader(f, delimiter='\t')
        for line in reader:
            data, ans = line
            data_list.append(data)
            answer_list.append(int(ans))
    return data_list, answer_list


def to_corpus(sentence: str, word_to_id: dict, id_to_word: dict) -> tuple:
    tagged_list = mecab.share(sentence)

    if word_to_id is None:
        word_to_id = {}
    if id_to_word is None:
        id_to_word = {}

    for word in tagged_list:
        if 'BOS/EOS' == word['class']:
            continue
        surface = word['surface']
        if surface not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[surface] = new_id
            id_to_word[new_id] = surface

    corpus = np.array([word_to_id[t['surface']] for t in tagged_list[1:-1]])
    return corpus, word_to_id, id_to_word


def to_squared(data_list, depth, dtype=np.int32):
    size = len(data_list)
    ret = np.zeros((size, depth), dtype=dtype)
    for i in range(size):
        w = min(len(data_list[i]), depth)
        ret[i, 0:w] = data_list[i][::]
    return ret


def to_one_hot(data_list):
    depth = np.max(data_list) + 1
    size = len(data_list)
    oh_list = np.zeros((size, depth), dtype=np.int32)

    for idx, data_id in enumerate(data_list):
        oh_list[idx, data_id] = 1

    return oh_list


