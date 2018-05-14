#! /usr/bin/python3
# coding:utf-8

import tensorflow as tf
import numpy as np

from lstm import conv_lstm_cell, init_state


def multi_layer_lstm(
    x, cs, hs, out_channels,
    filter_size=3, trainable_biases=False, forget_bias=0.9999
):
    """
    使用 tf.while_loop 实现的不限长度的 多层 Conv-LSTM 。

    其中输入 x 的第二轴(下标为 1 )为动态读取的循环次数(time-steps)，
    并且x的最后一个轴(通道轴)的 shape 必须是确定的，因为卷积参数的定义需要用到，
    而变量定义必须是确定的。

    cs 和 hs 是多层LSTM的 state 输入，他们分别是把每一层LSTM的 state 按通道轴拼接起来，
    并且通道轴的 shape 必须是确定的，通道轴的值均为所有LSTM层的输出通道数的和。
    由于其中不含池化，因此 feed 数据时，它们的宽和高轴的 shape 必须和x的宽高相等。
    如果不想使用带 state 接口的多层 Conv-LSTM，则传入 None

    out_channels 是多层 Conv-LSTM 的输出通道数列表，必须为一个以 int 数据为元素的 list。
    它必须是确定的数值。

    filter_size 是 Conv-LSTM 中的卷积核大小，是一个 int 数据，常规值为3
    可更改，但必须是确定的数值。

    trainable_biases 表示在 Conv-LSTM 中，是否每一个门控都有可训练偏置项。

    forget_bias 如果 trainable_biases 为 False ，那么，该参数将生效，
    该参数仅仅加入到遗忘门控上。

    Returns:

        cs, hs: 为经过LSTM网络后的所有 Conv-LSTM 层的 state 输出值。

        outs: 为所有 time-step 的输出按第二个轴(下表为 1 )拼接起来的的总输出，和输入类似。
    """
    with tf.variable_scope('Conv-LSTM') as scope:
        in_c = x.get_shape().as_list()[-1]
        # 创建 Conv-LSTM 共享参数， 然后在循环体重共享
        for j in range(len(out_channels)):
            if j != 0:
                in_c = out_channels[j-1]
            out_c = out_channels[j]
            filter_shape = [filter_size]*2 + [in_c+out_c, 4*out_c]
            with tf.variable_scope('body'):
                with tf.variable_scope('layer_%d' % j, reuse=False):
                    with tf.device('/cpu:0'):
                        weights = tf.get_variable(
                            'weights', shape=filter_shape, dtype=tf.float32,
                            initializer=tf.truncated_normal_initializer(stddev=0.1),
                            collections=['weights', tf.GraphKeys.GLOBAL_VARIABLES]
                        )
                        if trainable_biases:
                            biases = tf.get_variable(
                                'biases', shape=[filter_shape[-1]], dtype=tf.float32,
                                initializer=tf.constant_initializer(value=0.01),
                                collections=['biases', tf.GraphKeys.GLOBAL_VARIABLES]
                            )

        # 循环体
        def body(i, cs, hs, outs):
            with tf.variable_scope('body'):
                cell_in = x[:, i, ...]
                rank = len(cell_in.get_shape().as_list())
                c_list = tf.split(cs, out_channels, axis=rank-1)
                h_list = tf.split(hs, out_channels, axis=rank-1)
                for j in range(len(out_channels)):
                    c = c_list[j]
                    h = h_list[j]
                    with tf.variable_scope('layer_%d' % j, reuse=True):
                        c, h = conv_lstm_cell(
                            cell_in, (c, h), out_channels[j], filter_size=filter_size,
                            trainable_biases=trainable_biases,
                            forget_bias=forget_bias
                        )
                    cell_in = h
                    c_list[j] = c
                    h_list[j] = h
                i = i+1
                cs = tf.concat(c_list, axis=rank-1)
                hs = tf.concat(h_list, axis=rank-1)
                h_shape = tf.shape(h)
                batch_size = h_shape[0:1]
                img_shape = h_shape[1:]
                h_shape = tf.concat([batch_size, [1], img_shape], axis=0)
                h = tf.reshape(h, h_shape)
                outs = tf.concat([outs, h], axis=1)
                return i, cs, hs, outs

        # 循环前准备工作
        with tf.variable_scope('prepare') as scope:
            # 如果不想使用带 state 接口的多层 Conv-LSTM，则传入 None
            if cs is None and hs is None:
                cs = init_state(x[:, 0, ...], out_channels)
                hs = cs
            if cs is None:
                cs = init_state(x[:, 0, ...], out_channels)
            if hs is None:
                hs = init_state(x[:, 0, ...], out_channels)
            x_shape = tf.shape(x)
            batch_size = tf.strided_slice(x_shape, [0], [1], [1], name='batch-size')
            time_step = tf.identity(x_shape[1], name='time-steps')
            img_shape = tf.strided_slice(x_shape, [2], [-1], [1], name='video-resolution')
            outs_init_shape = tf.concat(
                [batch_size, [0], img_shape, [out_channels[-1]]], axis=0,
                name='outs-init-shape'
            )
            outs = tf.zeros(outs_init_shape, name='init-outs')
        i = tf.constant(0, dtype=tf.int32, name='loop-variable')  # 循环变量
        _, cs, hs, outs = tf.while_loop(
            lambda i, cs, hs, outs: tf.less(i, time_step),
            body, loop_vars=[i, cs, hs, outs],
            shape_invariants=[
                i.get_shape(), cs.get_shape(), hs.get_shape(),
                tf.TensorShape([None]*4+[out_channels[-1]])
            ], swap_memory=True,
            name='loop'
        )
    return cs, hs, outs


def main():
    sess = tf.Session()
    out_channels = [32, 64]
    x = tf.placeholder(tf.float32, shape=[None]*4+[3], name='x')
    x_ = np.ones([32, 5, 120, 160, 3], dtype=np.float32)
    cs = tf.placeholder(dtype=tf.float32, shape=[None]*3+[32+64], name='cs')
    hs = tf.placeholder(dtype=tf.float32, shape=[None]*3+[32+64], name='hs')
    cs_, hs_, outs = multi_layer_lstm(
        x, cs, hs, out_channels, trainable_biases=True
    )
    print(cs_)
    print(hs_)
    print(outs)

    init_op = tf.variables_initializer(tf.trainable_variables())
    sess.run(init_op)

    saver = tf.train.Saver()
    file_writer = tf.summary.FileWriter('logs', graph=sess.graph)
    summary = tf.summary.merge_all()
    saver.save(sess, 'logs/dynamic_lstm.ckpt')

    feed_dict = {
        x: x_,
        cs: np.zeros([32, 120, 160, 96]),
        hs: np.zeros([32, 120, 160, 96])
    }
    # outs_ = sess.run(outs, feed_dict=feed_dict)
    # print(outs_.shape)

    sess.close()
    file_writer.close()


if __name__ == '__main__':
    main()
