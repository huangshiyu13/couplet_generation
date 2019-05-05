#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: Shiyu Huang
@contact: huangsy13@gmail.com
@file: model_HMM.py
"""


class XiaHiddenVariable:
    def __init__(self, xia_word, log_prob, sententense):
        self.xia_word = xia_word
        self.log_prob = log_prob
        self.sententense = sententense


class MODEL_HMM:
    def __init__(self, unigram, transition, emit, keep_size):
        self.unigram = unigram
        self.transition = transition
        self.emit = emit
        self.keep_size = keep_size

    def set(self, keep_size):
        self.keep_size = keep_size

    def extend(self, hidden_varibale, word_now):
        sampled_next_words = set(self.transition[hidden_varibale.xia_word].samples()) | set(
            self.emit[word_now].samples())
        if len(sampled_next_words) == 0:
            sampled_next_words = self.unigram.samples()

        single_extend = []

        for next_word in sampled_next_words:
            log_prob = self.transition[hidden_varibale.xia_word].logprob(next_word) \
                       + self.emit[next_word].logprob(word_now) \
                       + hidden_varibale.log_prob
            single_extend.append(XiaHiddenVariable(next_word, log_prob, hidden_varibale.sententense + next_word))

        return sorted(single_extend, key=lambda x: x.log_prob, reverse=True)[:self.keep_size]

    def test(self, sentense):
        sentense_new = []
        for i in range(len(sentense)):
            sentense_new.append(sentense[i])
        HiddenVariables = [XiaHiddenVariable('<s>', 0., '')]
        for word_now in sentense_new:
            HiddenVariables_next = []
            for HiddenVariable in HiddenVariables:
                HiddenVariables_next += self.extend(HiddenVariable, word_now)
            HiddenVariables = list(
                sorted(HiddenVariables_next, key=lambda x: x.log_prob, reverse=True)[:self.keep_size])
        results = []

        for HiddenVariable in HiddenVariables:
            results.append((HiddenVariable.sententense, HiddenVariable.log_prob))
        return results
