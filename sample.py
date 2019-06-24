# -*- coding: utf-8 -*-
import lang.mecab as mecab


test_sentence = '踏切フリーダイヤルにて「自動車が逆走してしゃ断機を損傷した」連絡。信号係員が現地確認したところWB3のウエイト部分及びバーホルダーが損傷しているがしゃ断機本体に異常は見当たらず、しゃ断状態も正常である事を確認。念の為夜間にて予備品と交換した。'


def main1():
    mecab.test_mecab(test_sentence)


def sample_mecab():
    result = mecab.share(test_sentence)

    for token in result[1:-1]:
        print(token['surface'], token['class'])
