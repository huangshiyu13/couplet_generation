#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Shiyu Huang
# @File    : debug.py

import dataflow
import numpy as np
import tensorflow as tf
import time
from utils import check_dir, del_dir_under
import random
import sys
import os

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('please input GPU index')
        exit()

    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]

    if check_dir(dataflow.lstm_log_dir):
        del_dir_under(dataflow.lstm_log_dir)

    sess = tf.Session()
    writer = tf.summary.FileWriter(dataflow.lstm_log_dir, sess.graph)
    step = 0
    while True:
        print(step)
        step += 1
        time.sleep(2)
        bleu_score = random.random()
        eval_summary = tf.Summary(value=[tf.Summary.Value(
            tag='bleu', simple_value=bleu_score)])
        writer.add_summary(eval_summary, step)
        writer.flush()
