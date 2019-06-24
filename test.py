# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

import lang.util as lang_util
from common.rnn.optimizer import SGD

from model import SimpleRnn


def main():
    data_list, answer_list = lang_util.load_data('resources/base_data.tsv')

    word_to_id = {}
    id_to_word = {}
    corpus_list = []
    max_length = 0
    for data in data_list:
        corpus, word_to_id, id_to_word = lang_util.to_corpus(data, word_to_id, id_to_word)
        max_length = max(max_length, len(corpus))
        corpus_list.append(corpus)

    corpus_list = lang_util.to_squared(corpus_list, max_length, dtype=np.int32)
    answer_list = lang_util.to_one_hot(answer_list)

    vocab_size = len(word_to_id)
    embedding_size = 100
    hidden_size = 100
    output_size = answer_list.shape[1]

    print('data_list size:', len(data_list))     # -> 100
    print('word_to_id size:', len(word_to_id))   # -> 1217
    print()
    print('corpus_list.shape:', corpus_list.shape)
    print('answer_list.shape:', answer_list.shape)
    print()
    print('vocab_size:', vocab_size)
    print('embedding_size:', embedding_size)
    print('hidden_size:', hidden_size)
    print('output_size:', output_size)
    print()

    learning_rate = 0.1
    model = SimpleRnn(vocab_size, embedding_size, hidden_size, output_size)
    optimizer = SGD(learning_rate)

    max_epoch = 1

    loss_list = []
    for i in range(max_epoch):
        loss = model.forward(corpus_list, answer_list)
        model.backward()

        optimizer.update(model.params, model.grads)

        loss_list.append(loss)

    x = np.arange(len(loss_list))
    plt.plot(x, loss_list, marker='-')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.show()


def show_answer_histogram(answer_list):
    bc = np.bincount(answer_list)
    x = set(answer_list)
    plt.bar(list(x), bc)
    plt.xlabel('accident id')
    plt.ylabel('count')
    plt.show()


def main1():
    ary = [1,2,3,4,65,1,2,3,15]
    oh_list = lang_util.to_one_hot(ary)

    for oh in oh_list:
        print(oh)


if __name__ == '__main__':
    main()
