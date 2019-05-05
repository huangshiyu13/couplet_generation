#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Shiyu Huang 
@contact: huangsy13@gmail.com
@file: train_MEMM.py
"""

import pickle
from dataflow import data_save_path, save_base_dir, vocabular, MEMM_save_dir, load_unigram
from nltk.probability import FreqDist, ELEProbDist, ConditionalFreqDist, ConditionalProbDist
from utils import check_dir, check_file


def save_MEMM(duilians, v_size):
    bigram = []
    for duilian in duilians:
        shanglian = duilian[0]
        xialian = duilian[1]
        bigram += [((shang_duiying, xia_qian), xia_hou) for shang_duiying, xia_qian, xia_hou in
                   zip(shanglian[1:], xialian, xialian[1:])]
    ngram = ConditionalProbDist(ConditionalFreqDist(bigram), ELEProbDist, v_size)
    with open(MEMM_save_dir + 'memm.pkl', 'wb') as f:
        pickle.dump(ngram, f)


if __name__ == '__main__':
    assert check_dir(MEMM_save_dir)
    assert check_file(data_save_path)

    print('loading data...')
    with open(data_save_path, 'rb') as f:
        duilians = pickle.load(f)

    unigram = load_unigram()

    vocab_size = len(unigram.samples())
    print('vocabulary size: {}'.format(vocab_size))

    print('getting MEMM probability...')
    save_MEMM(duilians, vocab_size)
