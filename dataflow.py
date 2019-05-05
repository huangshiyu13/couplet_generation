#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Shiyu Huang 
@contact: huangsy13@gmail.com
@file: dataflow.py
"""

import pickle
from nltk.probability import ELEProbDist, FreqDist
import numpy as np

data_path = './data/'
save_base_dir = './data/'
data_save_path = save_base_dir + 'data.pkl'

model_path = './weights/'
vocabular = model_path + 'vocabs'
HMM_save_dir = model_path
MEMM_save_dir = model_path

hidden_unit = 1024
layers = 4
dropout = 0.2
batch_size = 32
learning_rate = 0.00001
lstm_save_dir = model_path
lstm_log_dir = model_path
init_path = lstm_save_dir + 'lstm.pkl'


def save_unigram(words):
    unigram = ELEProbDist(FreqDist(words))
    with open(save_base_dir + 'unigram.pkl', 'wb') as f:
        pickle.dump(unigram, f)
    return unigram


def load_unigram():
    with open(save_base_dir + 'unigram.pkl', 'rb') as f:
        unigram = pickle.load(f)
    return unigram


def get_lines(input_file):
    with open(input_file, 'r', encoding="utf-8") as file:
        lines = file.readlines()
    new_lines = []
    for line in lines:
        line = line.strip()
        line = line.split(' ')
        line = ['<s>'] + line + ['</s>']
        new_lines.append(line)
    return new_lines


def prepare_traininig_data():
    train_shanglian = data_path + 'train/in.txt'
    train_xialian = data_path + 'train/out.txt'

    shang_lines = get_lines(train_shanglian)
    xia_lines = get_lines(train_xialian)

    test_shanglian = data_path + 'test/in.txt'
    test_xialian = data_path + 'test/out.txt'

    test_shang_lines = get_lines(test_shanglian)
    test_xia_lines = get_lines(test_xialian)

    shang_lines += test_shang_lines
    xia_lines += test_xia_lines
    assert len(shang_lines) == len(xia_lines)
    duilians = []

    for duilian in zip(shang_lines, xia_lines):
        assert len(duilian[0]) == len(duilian[1])
        duilians.append(duilian)

    with open(data_save_path, 'wb') as f:
        pickle.dump(duilians, f)

    words = []
    with open(vocabular, 'r', encoding="utf-8") as f:
        vs = f.readlines()
        for line in vs:
            words.append(line.strip())
    for duilian in duilians:
        words += duilian[0] + duilian[1]
    save_unigram(words)


def read_vocab(vocab_file):
    f = open(vocab_file, 'rb')
    vocabs = [line.decode('utf8')[:-1] for line in f]
    f.close()
    return vocabs


def encode_text(words, vocab_indices):
    return [vocab_indices[word] for word in words if word in vocab_indices]


def padding_seq(seq):
    results = []
    max_len = 0
    for s in seq:
        if max_len < len(s):
            max_len = len(s)
    for i in range(0, len(seq)):
        l = max_len - len(seq[i])
        results.append(seq[i] + [0 for _ in range(l)])
    return results


def decode_text(labels, vocabs, end_token='</s>'):
    results = []
    for idx in labels:
        word = vocabs[idx]
        if word == end_token:
            return ' '.join(results)
        results.append(word)
    return ' '.join(results)


class DataFlow(object):
    def __init__(self, batch_size, data_dir, shuffle=True):
        self.batch_size = batch_size
        self.max_len = 50
        self.end_token = '</s>'
        self.shuffle = shuffle
        self.vocabs = read_vocab(vocabular)
        self.vocab_size = len(self.vocabs)
        self.vocab_indices = dict((c, i) for i, c in enumerate(self.vocabs))
        self.data = self.load_data(input_file=data_dir + 'in.txt', target_file=data_dir + 'out.txt')

        self.data_number = len(self.data)
        self.inds = self.shuffle_data()
        print('vocab_size: {} Total data:{}'.format(self.vocab_size, self.inds.shape[0]))
        self.idx = 0

    def shuffle_data(self):

        index = np.arange(self.data_number)
        if self.shuffle:
            np.random.shuffle(index)
        return index

    def load_data(self, input_file, target_file):
        data_all = []

        input_f = open(input_file, 'rb')
        target_f = open(target_file, 'rb')

        max_debug_num = -1
        debug_index_now = 0
        for input_line in input_f:
            debug_index_now += 1
            if max_debug_num > 0 and debug_index_now > max_debug_num:
                break
            input_line = input_line.decode('utf-8')[:-1]
            target_line = target_f.readline().decode('utf-8')[:-1]
            input_words = [x for x in input_line.split(' ') if x != '']
            if len(input_words) >= self.max_len:
                input_words = input_words[:self.max_len - 1]
            input_words.append(self.end_token)
            target_words = [x for x in target_line.split(' ') if x != '']
            if len(target_words) >= self.max_len:
                target_words = target_words[:self.max_len - 1]
            target_words = ['<s>', ] + target_words
            target_words.append(self.end_token)
            in_seq = encode_text(input_words, self.vocab_indices)
            target_seq = encode_text(target_words, self.vocab_indices)

            data_all.append({
                'in_seq': in_seq,
                'in_seq_len': len(in_seq),
                'target_seq': target_seq,
                'target_seq_len': len(target_seq) - 1
            })
        input_f.close()
        target_f.close()
        return data_all

    def prepare_data(self):
        if self.idx >= self.inds.shape[0] - self.batch_size:
            self.inds = self.shuffle_data()
            self.idx = 0

        in_seq = []
        target_seq = []
        in_seq_len = []
        target_seq_len = []

        for i in range(self.batch_size):
            index_now = self.inds[self.idx + i]
            # print(self.idx, index_now)
            in0 = self.data[index_now]['in_seq']
            target0 = self.data[index_now]['target_seq']
            in0_len = self.data[index_now]['in_seq_len']
            target0_len = self.data[index_now]['target_seq_len']

            in_seq.append(in0)
            target_seq.append(target0)
            in_seq_len.append(in0_len)
            target_seq_len.append(target0_len)

        in_seq = padding_seq(in_seq)
        target_seq = padding_seq(target_seq)
        in_seq = np.vstack(in_seq)
        in_seq_len = np.array(in_seq_len)
        target_seq = np.vstack(target_seq)
        target_seq_len = np.array(target_seq_len)

        self.idx += self.batch_size

        return in_seq, in_seq_len, target_seq, target_seq_len
