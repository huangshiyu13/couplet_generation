#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Shiyu Huang 
@contact: huangsy13@gmail.com
@file: demo_HMM.py
"""
from model_HMM import MODEL_HMM
from dataflow import HMM_save_dir, model_path
import pickle

sentenses = ['重阳节需要登高山', '桃李争春', '暖春天地宽']

if __name__ == '__main__':

    with open(model_path + 'unigram.pkl', 'rb') as f:
        unigram = pickle.load(f)
    with open(model_path + 'transition.pkl', 'rb') as f:
        transition = pickle.load(f)
    with open(model_path + 'emit.pkl', 'rb') as f:
        emit = pickle.load(f)
    keep_size = 40
    model = MODEL_HMM(unigram, transition, emit, keep_size=keep_size)

    for sentense in sentenses:
        results = model.test(sentense)
        print(sentense, results[0][0])
