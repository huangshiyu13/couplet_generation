#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Shiyu Huang
# @File    : model_MEMM.py


class XiaHiddenVariable:
    def __init__(self, xia_word, log_prob, sententense):
        self.xia_word = xia_word
        self.log_prob = log_prob
        self.sententense = sententense


class MODEL_MEMM:
    def __init__(self, unigram, MEMM_pro, keep_size):
        self.unigram = unigram
        self.MEMM_pro = MEMM_pro
        self.keep_size = keep_size

    def set(self, keep_size):
        self.keep_size = keep_size

    def extend(self, hidden_varibale, shang_word_now, xia_word_pre):
        sampled_next_words = set(self.MEMM_pro[(shang_word_now, xia_word_pre)].samples())
        if len(sampled_next_words) == 0:
            sampled_next_words = self.unigram.samples()

        single_extend = []

        for next_word in sampled_next_words:
            log_prob = self.MEMM_pro[(shang_word_now, xia_word_pre)].logprob(next_word) + hidden_varibale.log_prob
            single_extend.append(XiaHiddenVariable(next_word, log_prob, hidden_varibale.sententense + next_word))

        return sorted(single_extend, key=lambda x: x.log_prob, reverse=True)[:self.keep_size]

    def test(self, sentense):
        sentense_new = []
        for i in range(len(sentense)):
            sentense_new.append(sentense[i])
        HiddenVariables = [XiaHiddenVariable('<s>', 0., '')]
        for shang_word_now in sentense_new:
            HiddenVariables_next = []
            for HiddenVariable in HiddenVariables:
                xia_word_pre = HiddenVariable.xia_word

                HiddenVariables_next += self.extend(HiddenVariable, shang_word_now, xia_word_pre)
            HiddenVariables = list(
                sorted(HiddenVariables_next, key=lambda x: x.log_prob, reverse=True)[:self.keep_size])
        results = []

        for HiddenVariable in HiddenVariables:
            results.append((HiddenVariable.sententense, HiddenVariable.log_prob))
        return results
