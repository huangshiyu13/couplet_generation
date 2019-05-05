#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Shiyu Huang 
@contact: huangsy13@gmail.com
@file: demo_MEMM.py
"""
from model_MEMM import MODEL_MEMM
from dataflow import MEMM_save_dir, model_path
import pickle

sentenses = ['重阳节需要登高山', '桃李争春', '暖春天地宽']

if __name__ == '__main__':
    with open(model_path + 'unigram.pkl', 'rb') as f:
        unigram = pickle.load(f)
    with open(model_path + 'memm.pkl', 'rb') as f:
        MEMM_pro = pickle.load(f)
    keep_size = 10

    model = MODEL_MEMM(unigram, MEMM_pro, keep_size=keep_size)

    for sentense in sentenses:
        results = model.test(sentense)
        print(sentense, results[0][0])
