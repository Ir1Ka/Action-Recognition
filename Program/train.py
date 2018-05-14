#! /usr/bin/python3
# coding:utf-8

import tensorflow as tf
import numpy as np
from math import ceil
import sys
from concurrent.futures import ThreadPoolExecutor

import tools
from input_data import batchGenerator, trainBatchGeneratorThread
from conv import cnn, resnet, max_pool
from lstm import multi_layer_lstm, init_state


class pack_func():
    """
    用于解决循环内使用lambda表达式引用循环变量时的局部变量泄露问题
    """

    def __init__(self, data):
        self.data = data

    def get_data(self):
        return self.data


def inference(
    x, keep_prob, cs, hs, lstm_out_channels=None, cls_num=10
):

    if lstm_out_channels is None:
        lstm_out_channels = [32, 64]
    fc_units = 256
    layers = [x]

    # 多层CNN：四个卷积层，两个 2x2 池化层，输出通道数为 [16, 16, 32, 32]
    with tf.variable_scope('CNN'):
        time_steps = tf.shape(layers[-1])[1]
        # CNN输入需要是四维张量，而数据输入是五维张量，因此需要reshape
        in_c = layers[-1].get_shape().as_list()[-1]
        shape = tf.concat([[-1], tf.shape(layers[-1])[-3:-1], [in_c]], 0)
        reshape = tf.reshape(layers[-1], shape, name='cnn_in')
        layers.append(reshape)
        cnn_out = cnn(layers[-1])
        layers.append(cnn_out)
        # 还原为原始张量维数
        cnn_out_channel = layers[-1].get_shape().as_list()[-1]
        shape = tf.concat(
            [[-1, time_steps], tf.shape(layers[-1])[1:-1], [cnn_out_channel]], 0
        )
        reshape = tf.reshape(layers[-1], shape, name='restore_shape')
        layers.append(reshape)

    # 多层LSTM：两层不限长 Conv-LSTM
    with tf.variable_scope('LSTM'):
        cs, hs, outs = multi_layer_lstm(
            layers[-1], cs, hs, lstm_out_channels, trainable_biases=True
        )
        layers.append(outs)

    with tf.variable_scope('out_layers') as scope:
        # 取最后一个 time-step 的输出
        layers.append(layers[-1][:, -1, ...])
        in_c = layers[-1].get_shape().as_list()[-1]

        # TODO 此处待做动态优化
        pixel_size = (96 // 4) * (128 // 4) * in_c  # 需要固定

        with tf.variable_scope('local1'):
            reshape = tf.reshape(layers[-1], shape=[-1, pixel_size])
            layers.append(reshape)
            with tf.device('/cpu:0'):
                weights = tf.get_variable(
                    'weights', shape=[pixel_size, fc_units], dtype=tf.float32,
                    initializer=tf.truncated_normal_initializer(stddev=0.1),
                    collections=['weights', tf.GraphKeys.GLOBAL_VARIABLES]
                )
                biases = tf.get_variable(
                    'biases', shape=[fc_units], dtype=tf.float32,
                    initializer=tf.constant_initializer(value=0.01),
                    collections=['biases', tf.GraphKeys.GLOBAL_VARIABLES]
                )
            local1 = tf.nn.relu(tf.matmul(layers[-1], weights) + biases)
            layers.append(local1)
            drop = tf.nn.dropout(layers[-1], keep_prob)
            layers.append(drop)

        with tf.variable_scope('local2'):
            with tf.device('/cpu:0'):
                weights = tf.get_variable(
                    'weights', shape=[fc_units, cls_num], dtype=tf.float32,
                    initializer=tf.truncated_normal_initializer(stddev=0.1),
                    collections=['weights', tf.GraphKeys.GLOBAL_VARIABLES]
                )
                biases = tf.get_variable(
                    'biases', shape=[cls_num], dtype=tf.float32,
                    initializer=tf.constant_initializer(value=0.01),
                    collections=['biases', tf.GraphKeys.GLOBAL_VARIABLES]
                )
            local2 = tf.matmul(layers[-1], weights) + biases
            layers.append(local2)
    logits = layers[-1]
    return logits, cs, hs


class trainer():
    def __init__(self, is_recog=False):
        self.batch_generator = batchGenerator()

        # 一些超参数
        self.lstm_out_channels = [32, 64]
        self.cls_num = len(self.batch_generator.cls_list)
        self.init_lr = 1e-3
        self.train_report_rate = 100
        self.test_report_rate = 1000
        self.max_gradient = 1.

        self.test_size = self.batch_generator.test_num
        self.train_size = self.batch_generator.data_num
        self.epochs = self.batch_generator.epochs
        self.batch_size = self.batch_generator.batch_size
        self.total_step = int(self.train_size/self.batch_size)*self.epochs
        self.height = self.batch_generator.height
        self.width = self.batch_generator.width
        self.channels = self.batch_generator.channels

        # placeholder
        self.x = tf.placeholder(
            tf.float32, [None, None, self.height, self.width, self.channels],
            name='x'
        )
        tf.summary.image('video_in', self.x[:, 0, ...], collections=['layers_out'])
        self.y = tf.placeholder(tf.int32, [None], name='y')
        self.keep_prob = tf.placeholder(tf.float32, [], name='keep_prob')
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.lr = tf.train.exponential_decay(
            self.init_lr, self.global_step, self.total_step, 0.001, name='auto_lr'
        )
        tf.summary.scalar('lr', self.lr, collections=['train'])
        self.placeholders = [self.x, self.y, self.keep_prob]

        self.is_recog = is_recog
        if not self.is_recog:
            # 开启 batch 生成器线程
            self.batch_generator_thread = trainBatchGeneratorThread(self.batch_generator)
            self.batch_generator_thread.start()
            self.test_pool = ThreadPoolExecutor(max_workers=1)

        self.gen_model()
        self.all_variables = tf.global_variables()+tf.local_variables()
        self.init_op = tf.variables_initializer(self.all_variables, name='init-op')

        self.sess = tf.Session()

        self.logs_dir = 'data/logs'
        self.model_fname = 'data/logs/model.ckpt'

        # 一些TensorBorad的op
        self.saver = tf.train.Saver()
        if not self.is_recog:
            self.summary_writer = tf.summary.FileWriter(self.logs_dir, graph=self.sess.graph)
            self.train_merged = tf.summary.merge_all(key='train')
            # 以下的 merge 只在测试的时候用
            self.test_merged = tf.summary.merge_all(key='test')
            self.layers_out_merged = tf.summary.merge_all(key='layers_out')

    def feed(self, values):
        length = len(self.placeholders)
        if length != len(values):
            raise ValueError('''values 必须和 placeholders 的长度相等''')
        feed_dict = {}
        for i in range(length):
            feed_dict[self.placeholders[i]] = values[i]
        return feed_dict

    def gen_model(self):
        self.logits, _, _ = inference(
            self.x, self.keep_prob, None, None,
            lstm_out_channels=self.lstm_out_channels,
            cls_num=self.cls_num
        )
        self.tvars = tf.trainable_variables()
        # print('可训练参数个数：', len(self.tvars))
        self.weights = tf.get_collection('weights')
        with tf.variable_scope('regul'):
            # L1正则化
            # l1 = tf.contrib.layers.l1_regularizer(1e-4)
            # l1_ = [l1(weight) for weight in self.weights]
            # L2正则化
            l2 = tf.contrib.layers.l2_regularizer(1e-4)
            l2_ = [l2(weight) for weight in self.weights]
            l2__ = tf.add_n(l2_, name='l2')
        with tf.variable_scope('loss'):
            self.xentropy = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=self.y, logits=self.logits
                ),
                name='xentropy'
            ) + l2__
            self.test_loss = tf.placeholder(tf.float32, [], name='test_loss_ph')
        tf.summary.scalar('train_loss', self.xentropy, collections=['train'])
        tf.summary.scalar('test_loss', self.test_loss, collections=['test'])
        # 梯度裁剪
        gradients = tf.gradients(self.xentropy, self.tvars)
        self.clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient)
        opt = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_op = opt.apply_gradients(
            zip(self.clipped_gradients, self.tvars),
            global_step=self.global_step, name='train_op'
        )
        with tf.variable_scope('prediction'):
            self.correct_prediction = tf.equal(
                tf.argmax(self.logits, 1), tf.cast(self.y, dtype=tf.int64)
            )
            # 准确率
            self.acc = tf.reduce_mean(
                tf.cast(self.correct_prediction, dtype=tf.float32), name='acc'
            )
            self.test_acc = tf.placeholder(tf.float32, [], name='test_acc_ph')
        tf.summary.scalar('train_acc', self.acc, collections=['train'])
        tf.summary.scalar('test_acc', self.test_acc, collections=['test'])

    def train_a_step(self, global_step=0):
        batch, labels = self.batch_generator_thread.get_train_batch()
        if batch is None:
            # 训练结束
            return None

        is_summary = False
        ops = [self.train_op, self.xentropy, self.acc]
        if global_step % self.train_report_rate == 0:
            ops.append(self.train_merged)
            is_summary = True

        values = [batch, labels, 0.5]
        feed_dict = self.feed(values)
        r = self.sess.run(ops, feed_dict=feed_dict)
        if is_summary:
            self.summary_writer.add_summary(r[-1], global_step=global_step)

        return r[1:3]

    def test(self, global_step=None):
        self.test_pool.submit(self.batch_generator.test_batch_generator)
        correct = []
        total_loss = 0
        ops = [self.xentropy, self.correct_prediction]
        if global_step is not None:
            ops.append(self.layers_out_merged)
        while True:
            batch, labels = self.batch_generator.test_batchs.get(timeout=10)
            if batch is None:
                break
            batch_size = batch.shape[0]
            values = [batch, labels, 1.0]
            feed_dict = self.feed(values)
            r = self.sess.run(
                ops,
                feed_dict=feed_dict
            )
            total_loss += r[0]*batch_size
            correct.extend(r[1].tolist())
        acc = np.mean(correct)
        total_loss /= self.test_size
        if global_step is not None:
            t_merged = self.sess.run(
                self.test_merged,
                feed_dict={self.test_acc: acc, self.test_loss: total_loss}
            )
            self.summary_writer.add_summary(r[2], global_step=global_step)
            self.summary_writer.add_summary(t_merged, global_step=global_step)
        return acc, total_loss

    def test_with_save(self, global_step=None):
        if global_step is None:
            str_format = '\nFinally test:  '
        else:
            str_format = '\nTest:  '
        self.saver.save(self.sess, self.model_fname, global_step=global_step)
        acc, loss = self.test(global_step=global_step)
        str_format += 'acc = %6.4s, loss = %6.4s'
        print(str_format % (str(acc), str(loss)))

    def train(self, restore=True):

        if restore:
            self.saver.restore(self.sess, self.model_fname)
        else:
            self.sess.run(self.init_op)
        lr_ = self.init_lr
        step = 0
        print('总训练步：', self.total_step)
        try:
            while True:
                if step == self.total_step*0.5 or step == self.total_step*0.7 or step == self.total_step*0.9:
                    lr_ *= 0.1
                train_loss, acc_ = self.train_a_step(global_step=step)
                if train_loss is None:
                    break

                str_format = 'Train step %8sth:' % str(step) + 'acc = %6.4s, loss = %6.4s'
                print(str_format % (str(acc_), str(train_loss)), end='\r')
                if step % self.test_report_rate == 0:
                    self.test_with_save(global_step=step)
                step += 1
        finally:
            self.test_with_save(global_step=step)

    def recog(self, fname_or_list):
        """
        识别 demo，输入一个或多个视频文件路径，返回识别结果
        """
        if not isinstance(fname_or_list, list):
            fname_or_list = [fname_or_list]
        j = self.batch_generator.test_batch_size
        copies_num = ceil(len(fname_or_list) / j)
        logits_list = []
        for i in range(0, copies_num*j, j):
            fnames = fname_or_list[i: i+j]
            batch, _ = self.batch_generator.handle_batch(fnames, is_test=True)
            # 用于 feed 数据
            labels = [0] * len(batch)
            values = [batch, labels, 1.0]
            feed_dict = self.feed(values)
            logits = self.sess.run(self.logits, feed_dict=feed_dict)
            logits_list.append(logits)
        logits = np.concatenate(logits_list)
        probs = tools.softmax(logits)
        labels = np.argmax(logits, axis=-1)
        return probs, labels

    def close(self):
        self.sess.close()
        if not self.is_recog:
            self.summary_writer.close()
            self.test_pool.shutdown()
        self.batch_generator.pool.shutdown()


def main(restore=True):
    with tools.timer('build model'):
        train = trainer()
    with tools.timer('train'):
        train.train(restore=restore)
    train.close()


def parse_argv():
    restore = None
    argv = sys.argv
    if len(argv) >= 2:
        restore_arg = argv[1].split('=')
        if len(restore_arg) != 2:
            raise Exception('''参数需要使用 = 隔开''')
        if restore_arg[0] == '--restore':
            if restore_arg[1] == 'True':
                restore = True
            elif restore_arg[1] == 'False':
                restore = False
        else:
            raise('''参数名必须为 --restore''')
    else:
        restore = True
    if restore is None:
        raise Exception('''参数值必须为 True 或 False''')
    return restore


if __name__ == '__main__':
    restore = parse_argv()
    print('是否恢复参数：', restore)
    main(restore=restore)






