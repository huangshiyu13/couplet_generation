#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Shiyu Huang 
@contact: huangsy13@gmail.com
@file: train_HMM.py
"""

import pickle
from dataflow import data_save_path, HMM_save_dir, load_unigram
from nltk.probability import ELEProbDist, ConditionalFreqDist, ConditionalProbDist
from utils import check_dir, check_file


def save_trainsition(duilians, v_size):
    bigram = []
    for duilian in duilians:
        for duilian_sub in duilian:
            bigram += [(w1, w2) for w1, w2 in zip(duilian_sub, duilian_sub[1:])]
    ngram = ConditionalProbDist(ConditionalFreqDist(bigram), ELEProbDist, v_size)
    with open(HMM_save_dir + 'transition.pkl', 'wb') as f:
        pickle.dump(ngram, f)


def save_emit(duilians, v_size):
    bigram = []
    for duilian in duilians:
        shanglian = duilian[0]
        xialian = duilian[1]
        bigram += [(w1, w2) for w1, w2 in zip(xialian, shanglian)]
    ngram = ConditionalProbDist(ConditionalFreqDist(bigram), ELEProbDist, v_size)
    with open(HMM_save_dir + 'emit.pkl', 'wb') as f:
        pickle.dump(ngram, f)


if __name__ == '__main__':
    assert check_dir(HMM_save_dir)
    assert check_file(data_save_path)

    print('loading data...')
    with open(data_save_path, 'rb') as f:
        duilians = pickle.load(f)
    unigram = load_unigram()

    vocab_size = len(unigram.samples())
    print('vocabulary size: {}'.format(vocab_size))

    print('getting the transition probability of the data...')
    save_trainsition(duilians, vocab_size)

    print('getting the emit probability of the data...')
    save_emit(duilians, vocab_size)
