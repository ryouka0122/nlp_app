#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function
import MeCab
import sys


def share(sentence: str) -> list:
    tagger = MeCab.Tagger('')
    m = tagger.parseToNode(sentence)
    word_list = []

    while m:
        infos = m.feature.split(',')
        word = {
            'surface': m.surface,
            'class': infos[0],
            'class_detail1': infos[1],
            'class_detail2': infos[2],
            'class_detail3': infos[3],
            'conj_type': infos[4],
            'conj_from': infos[5],
            'origin': infos[6],
            'katakana': m.surface,
            'pron': m.surface
        }
        if len(infos) > 7:
            word.update({
                'katakana': infos[7],
                'pron': infos[8]
            })
        word_list.append(word)
        m = m.next

    return word_list








def test_mecab(sentence):

    try:
        print(MeCab.VERSION)

        t = MeCab.Tagger(" ".join(sys.argv))
        print(t.parse(sentence))

        m = t.parseToNode(sentence)

        while m:
            print(m.surface, "\t", m.feature)
            m = m.next
        print("EOS")

        lattice = MeCab.Lattice()
        t.parse(lattice)
        lattice.set_sentence(sentence)
        length = lattice.size()

        for i in range(length + 1):
            b = lattice.begin_nodes(i)
            e = lattice.end_nodes(i)

            while b:
                print("B[%d] %s\t%s" % (i, b.surface, b.feature))
                b = b.bnext

            while e:
                print("E[%d] %s\t%s" % (i, e.surface, e.feature))
                e = e.bnext
        print("EOS")

        d = t.dictionary_info()
        while d:
            print("filename: %s" % d.filename)
            print("charset: %s" % d.charset)
            print("size: %d" % d.size)
            print("type: %d" % d.type)
            print("lsize: %d" % d.lsize)
            print("rsize: %d" % d.rsize)
            print("version: %d" % d.version)
            d = d.next

    except RuntimeError as e:
        print("RuntimeError:", e)


if __name__ == '__main__':
    test_mecab()
