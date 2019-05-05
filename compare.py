#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Shiyu Huang
# @File    : compare.py
from model_HMM import MODEL_HMM
from model_MEMM import MODEL_MEMM
from dataflow import HMM_save_dir, save_base_dir, MEMM_save_dir
import pickle
import tensorflow as tf
import sys
import os
import model_lstm as model
import dataflow

sentenses = ['古今奇观属岩壑', '青山不墨千秋画', '两岸凉生菰叶雨', '春眠不觉晓', '无边落木萧萧下', '两只黄鹂鸣翠柳'
    , '万紫千红春无限', '深秋帘幕千家雨', '月透柳帘窥案卷']


def test_HMM(f_log):
    print('loading HMM model...')
    f_log.write('HMM:\n')
    with open(save_base_dir + 'unigram.pkl', 'rb') as f:
        unigram = pickle.load(f)
    with open(HMM_save_dir + 'transition.pkl', 'rb') as f:
        transition = pickle.load(f)
    with open(HMM_save_dir + 'emit.pkl', 'rb') as f:
        emit = pickle.load(f)
    keep_size = 40
    for sentense in sentenses:
        model = MODEL_HMM(unigram, transition, emit, keep_size=keep_size)
        results = model.test(sentense)
        print('{}\t{}'.format(sentense, results[0][0]))
        f_log.write('{}\t{}\n'.format(sentense, results[0][0]))


def test_MEMM(f_log):
    print('loading MEMM model...')
    f_log.write('MEMM:\n')
    with open(save_base_dir + 'unigram.pkl', 'rb') as f:
        unigram = pickle.load(f)
    with open(MEMM_save_dir + 'memm.pkl', 'rb') as f:
        MEMM_pro = pickle.load(f)
    keep_size = 40
    for sentense in sentenses:
        model = MODEL_MEMM(unigram, MEMM_pro, keep_size=keep_size)
        results = model.test(sentense)
        print('{}\t{}'.format(sentense, results[0][0]))
        f_log.write('{}\t{}\n'.format(sentense, results[0][0]))


def test_lstm(f_log):
    print('loading lstm model...')
    f_log.write('LSTM:\n')
    in_seq_holder = tf.placeholder(tf.int32, shape=[1, None], name='in_seq')
    in_seq_len_holder = tf.placeholder(tf.int32, shape=[1], name='in_seq_len')
    test_model = model.Seq2Seq()
    vocabs = dataflow.read_vocab(dataflow.vocabular)
    vocab_indices = dict((c, i) for i, c in
                         enumerate(vocabs))
    voc_size = len(vocabs)
    test_model.build_infer(in_seq_holder, in_seq_len_holder,
                           voc_size,
                           dataflow.hidden_unit, dataflow.layers, name_scope='infer')

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    saver = model.Saver(sess)
    saver.load(dataflow.init_path, scope_name='infer', del_scope=True)

    for sentense in sentenses:
        new_sentense = []
        for v in sentense:
            new_sentense.append(v)
        new_sentense.append('</s>')
        in_seq = dataflow.encode_text(new_sentense, vocab_indices)
        in_seq_len = len(in_seq)
        outputs = sess.run(test_model.infer_output,
                           feed_dict={
                               in_seq_holder: [in_seq],
                               in_seq_len_holder: [in_seq_len]})
        output = outputs[0]
        output = dataflow.decode_text(output, vocabs)
        output = ''.join(output.split(' '))
        print('{}\t{}'.format(sentense, output))
        f_log.write('{}\t{}\n'.format(sentense, output))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('please input GPU index')
        exit()
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]
    f = open('compare.txt', 'w')
    test_lstm(f)
    test_HMM(f)
    test_MEMM(f)
    f.close()
