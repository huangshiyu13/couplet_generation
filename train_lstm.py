#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Shiyu Huang
# @File    : train_lstm.py

import os
import time
import sys
import numpy as np
import tensorflow as tf
import dataflow
import threading
import queue
from utils import check_dir, create_dir, del_dir_under

from model_lstm import Saver
from dataflow import decode_text
import model_lstm as model
import bleu
import math

saveTime = 1000
print_time = 200
maxstep = 5000000
eval_time = 1000
start_step = 704800


def keep_xiaoshu(value, xiaoshuhou):
    tt = pow(10, xiaoshuhou)
    return int(value * tt) / float(tt)


class Producter(threading.Thread):
    def __init__(self, queue, data_loader, step):
        self.queue = queue
        threading.Thread.__init__(self)
        self.data_loader = data_loader
        self.step = step

    def run(self):
        for i in range(start_step, self.step + 100):
            data = self.data_loader.prepare_data()
            self.queue.put(data)


class Consumer(threading.Thread):
    def __init__(self, queue, step, pid, voc_size, valid_data_flow):
        self.queue = queue
        self.valid_data_flow = valid_data_flow
        threading.Thread.__init__(self)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        self.train_in_seq = tf.placeholder(tf.int32, shape=[dataflow.batch_size, None], name='in_seq')
        self.train_in_seq_len = tf.placeholder(tf.int32, shape=[dataflow.batch_size], name='in_seq_len')
        self.train_target_seq = tf.placeholder(tf.int32, shape=[dataflow.batch_size, None], name='target_seq')
        self.train_target_seq_len = tf.placeholder(tf.int32, shape=[dataflow.batch_size], name='target_seq_len')

        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        self.model = model.Seq2Seq()

        self.model.build(self.train_in_seq, self.train_in_seq_len, self.train_target_seq, self.train_target_seq_len,
                         voc_size,
                         dataflow.hidden_unit, dataflow.layers, dataflow.dropout, dataflow.learning_rate,
                         name_scope='train')
        self.model.build_infer(self.train_in_seq, self.train_in_seq_len,
                               voc_size,
                               dataflow.hidden_unit, dataflow.layers, name_scope='infer')

        self.transfer = model.transfer_params(from_scope='train', to_sope='infer')
        self.sess.run(tf.global_variables_initializer())
        self.saver = Saver(self.sess)
        if start_step == 1:
            continue_train = False
        else:
            continue_train = True
        self.saver.auto_save_init(save_dir=dataflow.lstm_save_dir, save_interval=saveTime, max_keep=5,
                                  scope_name='train', continue_train=continue_train)
        self.saver.load(dataflow.init_path, scope_name='train', del_scope=True)

        print('Training Begin')

        self.step = step
        print('pid:{}'.format(pid))
        self.pid = pid

        if start_step == 1:
            if check_dir(dataflow.lstm_log_dir):
                del_dir_under(dataflow.lstm_log_dir)
            else:
                create_dir(dataflow.lstm_log_dir)
        else:
            assert check_dir(dataflow.lstm_log_dir)

        self.writer = tf.summary.FileWriter(dataflow.lstm_log_dir, self.sess.graph)

    def run(self):
        train_loss = []

        for i in range(start_step, self.step + 1):
            # s = time.time()
            w_i = 0
            while q.empty():
                w_i += 1
                if w_i > 2000:
                    print('the queue is empty, wait for data')
                    time.sleep(1)
                    w_i = 0
                continue
            try:
                data = self.queue.get(1, 3)
            except:
                print('can\'t fetch data!')
                continue

            if i == start_step + 5:
                start_time = time.time()

            batch = data

            list_dict = [{self.train_in_seq: batch[0],
                          self.train_in_seq_len: batch[1],
                          self.train_target_seq: batch[2],
                          self.train_target_seq_len: batch[3]}]
            feed = {}
            for d in list_dict:
                feed.update(d)
            try:
                (_, train_loss_iter) = self.sess.run(
                    [self.model.train_op, self.model.loss],
                    feed_dict=feed)
            except:
                print('someting is wrong for traiing')
                self.saver.load(dataflow.init_path, scope_name='train', del_scope=True, show_names=True)
                continue

            if math.isnan(train_loss_iter) or math.isinf(train_loss_iter) or train_loss_iter > 500:
                print('loss:', train_loss_iter)
                self.saver.load(dataflow.init_path, scope_name='train', del_scope=True, show_names=True)
                continue

            train_loss.append(train_loss_iter)

            if i > start_step + 5 and i % print_time == 0:
                train_summary = tf.Summary(value=[tf.Summary.Value(
                    tag='loss', simple_value=np.mean(train_loss))])
                self.writer.add_summary(train_summary, i)
                self.writer.flush()
                print(self.pid, sys.argv[1], ' step :', i, 'time :{}s'.format(int(time.time() - start_time)),
                      'average time:{}s'.format(
                          keep_xiaoshu(value=(time.time() - start_time) / (i - start_step), xiaoshuhou=5)),
                      'ETS time:{}h'.format(
                          keep_xiaoshu(value=(time.time() - start_time) / (i - start_step) * (self.step - i) / 3600,
                                       xiaoshuhou=2)),
                      'loss :', np.mean(train_loss), 'l_r :', dataflow.learning_rate)
                train_loss = []

            if i % eval_time == 0:
                print('begin evaluation...')
                bleu_score = self.eval()
                print('eval score:{}'.format(bleu_score))
                eval_summary = tf.Summary(value=[tf.Summary.Value(
                    tag='bleu', simple_value=bleu_score)])
                self.writer.add_summary(eval_summary, i)
                self.writer.flush()
            self.saver.auto_save(i)

    def eval(self):
        target_results = []
        output_results = []
        self.sess.run(self.transfer)
        for _ in range(int(self.valid_data_flow.data_number / dataflow.batch_size)):
            batch = self.valid_data_flow.prepare_data()
            list_dict = [{self.train_in_seq: batch[0],
                          self.train_in_seq_len: batch[1]}
                         ]
            target_seq = batch[2]
            in_seq = batch[0]
            feed = {}
            for d in list_dict:
                feed.update(d)
            outputs = self.sess.run(self.model.infer_output, feed_dict=feed)
            for i in range(len(outputs)):
                output = outputs[i]
                target = target_seq[i]
                output_text = decode_text(output,
                                          self.valid_data_flow.vocabs).split(' ')
                target_text = decode_text(target[1:],
                                          self.valid_data_flow.vocabs).split(' ')

                target_results.append([target_text])
                output_results.append(output_text)
                if _ % 100 == 0 and i == 0:
                    print('====================')
                    input_text = decode_text(in_seq[i],
                                             self.valid_data_flow.vocabs)
                    print('src:' + input_text)
                    print('output: ' + ' '.join(output_text))
                    print('target: ' + ' '.join(target_text))
        return bleu.compute_bleu(target_results, output_results)[0] * 100


if __name__ == '__main__':

    pid = os.getpid()

    if not check_dir(dataflow.lstm_save_dir):
        create_dir(dataflow.lstm_save_dir)

    print('loading training data...')
    train_data_flow = dataflow.DataFlow(dataflow.batch_size, data_dir=dataflow.data_path + 'train/')
    print('loading evaluation data...')
    valid_data_flow = dataflow.DataFlow(dataflow.batch_size, data_dir=dataflow.data_path + 'test/', shuffle=True)

    q = queue.Queue(maxsize=100)

    pt = Producter(queue=q, data_loader=train_data_flow, step=maxstep)
    ce = Consumer(step=maxstep, queue=q, pid=pid, voc_size=train_data_flow.vocab_size,
                  valid_data_flow=valid_data_flow)
    pt.start()
    ce.start()
    pt.join()
    ce.join()
