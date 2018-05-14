#! /usr/bin/python3
# coding:utf-8

import tensorflow as tf
import numpy as np

import tools


def conv2d(
    x, filter_shape, stddev=0.1, biases_init=0.01, strides=None,
    padding='VALID', has_biases=True, name=None
):
    """
    含偏置的卷积操作，默认 `strides = [1, 1, 1, 1]`
    """
    if strides is None:
        strides = [1, 1, 1, 1]
    with tf.device('/cpu:0'):
        weights = tf.get_variable(
            'weights',
            shape=filter_shape,
            dtype=tf.float32,
            initializer=tf.truncated_normal_initializer(stddev=stddev),
            collections=['weights', tf.GraphKeys.GLOBAL_VARIABLES]
        )
    conv = tf.nn.conv2d(x, weights, strides, padding, name=name)
    if has_biases:
        with tf.device('/cpu:0'):
            biases = tf.get_variable(
                'biases', shape=[filter_shape[-1]],
                dtype=tf.float32,
                initializer=tf.constant_initializer(value=biases_init),
                collections=['biases', tf.GraphKeys.GLOBAL_VARIABLES]
            )
        conv = tf.nn.bias_add(conv, biases)
    return conv


def conv2d_relu(
    x, filter_shape, stddev=0.1, biases_init=0.1, strides=None,
    padding='VALID', has_biases=True, name=None
):
    """
    含偏置和relu的卷积，默认 `strides = [1, 1, 1, 1]`
    """
    conv0 = conv2d(
        x, filter_shape, stddev=stddev, biases_init=biases_init, strides=strides,
        padding=padding, has_biases=has_biases, name=name
    )
    return tf.nn.relu(conv0)


def max_pool(inputs, pool_size=2, padding='VALID', name=None):
    """
    最大池化操作，默认为2×2池化，一般name参数使用默认值即可，方法内将自动命名
    """
    if name is None:
        name = 'max_pool%dx%d' % (pool_size, pool_size)
    ksize_strids = [1, pool_size, pool_size, 1]
    return tf.nn.max_pool(inputs, ksize_strids, ksize_strids, padding, name=name)


def res_block_1x1_3x3_1x1(x, in_channels, out_channel):
    """
    ResNet的块结构，由两个1×1、一个3×3和一个快捷链接构成。

    Args:

        x: Block的输入

        in_channels: Block的输入通道数

        out_channel: Block的输出通道数

    Return:

        y: Block的输出
    """
    # 输入和输出通道数只能是相等或者二倍关系
    twice = None
    if in_channels == out_channel:
        twice = False
    elif 2*in_channels == out_channel:
        twice = True
    else:
        raise KeyError('残差块输入输出通道数不匹配。')

    layer = [x]
    # 1×1的卷积，通道数为输出通道数的1/4
    with tf.variable_scope('conv0_in_block'):
        conv1 = conv2d_relu(layer[-1], [1, 1, in_channels, in_channels//4])
        layer.append(conv1)
    # 3×3的卷积，通道数为输出通道数的1/4
    with tf.variable_scope('conv1_in_block'):
        conv2 = conv2d_relu(layer[-1], [3, 3, in_channels//4, in_channels//4])
        layer.append(conv2)

    # 1×1的卷积，通道数为输出通道数
    with tf.variable_scope('conv2_in_block'):
        conv3 = conv2d(layer[-1], [1, 1, in_channels//4, out_channel])
        layer.append(conv3)
    with tf.variable_scope('shortcut'):
        if twice is False:
            shortcut = layer[0]
        else:
            ndim = len(layer[0].get_shape().as_list())
            pad = [[0, 0]] * (ndim - 1) + [[0, in_channels]]
            shortcut = tf.pad(layer[0], pad)
    layer.append(layer[-1] + shortcut)
    relu = tf.nn.relu(layer[-1])
    layer.append(relu)
    return layer[-1]


def res_block(x, in_channels, out_channel):
    return res_block_1x1_3x3_1x1(x, in_channels, out_channel)


def resnet(x, out_channels=None, n=1, is_global_pool=True):
    """
    ResNet

    Args:

        x: 输入

        n:每个输出通道数所使用的残差块个数

        out_channels:残差块输出通道数，默认值为[64, 128, 256, 512]

        is_global_pool:是否在ResNet后加全局平均池化，即把图像部分的池化为一个值，默认True

    Returns:

        out:网络输出
    """
    if out_channels is None:
        out_channels = [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3
        n = 1
    if not isinstance(out_channels, (list, int)):
        raise ValueError('''输出通道必须是list或int''')

    # print(in_channel, out_channels)

    layer = [x]
    in_channel = layer[-1].get_shape().as_list()[-1]
    with tf.variable_scope('conv0'):
        conv0 = conv2d_relu(layer[-1], [7, 7, in_channel, out_channels[0]])
        layer.append(conv0)
        pool = max_pool(layer[-1])
        layer.append(pool)

    for index in range(len(out_channels)):
        for i in range(n):
            with tf.variable_scope('conv%d_%d' % (index+1, i)):
                if index != 0 and i == 0:
                    conv = res_block(layer[-1], out_channels[index-1], out_channels[index])
                else:
                    conv = res_block(layer[-1], out_channels[index], out_channels[index])
                layer.append(conv)

    with tf.variable_scope('finally_layer'):
        if is_global_pool:
            global_pool = tf.reduce_mean(layer[-1], [-2, -3], name='global_pool')
            layer.append(global_pool)
        # else:
            # 2×2的最大池化
            # layer.append(max_pool(layer[-1]))

    return layer[-1]


def cnn(x):
    """
    CNN网络
    """
    layers = [x]
    in_channel = x.get_shape().as_list()[-1]
    with tf.variable_scope('h_conv0'):
        h_conv = conv2d_relu(layers[-1], [3, 3, 3, 16], padding='SAME')
        layers.append(h_conv)
        layers.append(max_pool(layers[-1], padding='SAME'))
        tf.summary.image('h_conv0', layers[-1], collections=['layers_out'])

    with tf.variable_scope('h_conv0_1'):
        h_conv = conv2d_relu(layers[-1], [3, 3, 16, 16], padding='SAME')
        layers.append(h_conv)
        tf.summary.image('h_conv0_1', layers[-1], collections=['layers_out'])

    with tf.variable_scope('h_conv1'):
        h_conv = conv2d_relu(layers[-1], [3, 3, 16, 32], padding='SAME')
        layers.append(h_conv)
        layers.append(max_pool(layers[-1], padding='SAME'))
        tf.summary.image('h_conv1', layers[-1], collections=['layers_out'])

    with tf.variable_scope('h_conv1_1'):
        h_conv = conv2d_relu(layers[-1], [3, 3, 32, 32], padding='SAME')
        layers.append(h_conv)
        tf.summary.image('h_conv1_1', layers[-1], collections=['layers_out'])

    return layers[-1]


def main():
    import input_data
    x = tf.placeholder(tf.float32, [None, 120, 160, 3], 'x')
    y_ = tf.placeholder(tf.int32, [None], 'y_')
    resnet_out = resnet(x)
    shape = resnet_out.get_shape().as_list()
    with tf.variable_scope('fc'):
        fc_weights = tf.get_variable(
            'weights',
            shape=[shape[-1], 5],
            dtype=tf.float32,
            initializer=tf.truncated_normal_initializer(stddev=0.1),
            collections=['weights', tf.GraphKeys.GLOBAL_VARIABLES]
        )
        fc = tf.matmul(resnet_out, fc_weights)

    prediction = tf.arg_max(fc, -1)

    xentropy = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_, logits=fc)
    )
    train_step = tf.train.AdamOptimizer(1e-4).minimize(xentropy)

    train_variable = tf.trainable_variables()
    gra = tf.gradients(xentropy, train_variable)

    batch_generator_thread = input_data.trainBatchGeneratorThread()
    batch_generator_thread.start()

    sess = tf.Session()
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # batch, labels = batch_generator_thread.get_train_batch()
    # batch = batch[:, 0, ...]

    step = 0
    while True:
        print('train_step:', step)
        step += 1
        batch, labels = batch_generator_thread.get_train_batch()
        if batch is None:
            break
        print('batch.shape =', batch.shape)
        batch = batch[:, 0, ...]
        print('labels:', labels)
        # labels = np.eye(10)[labels]
        # print('batch:', batch[:, 60, 80, :])
        feed_dict = {x: batch, y_: labels}
        c_out, pre, logits, xen, _ = sess.run([resnet_out, prediction, fc, xentropy, train_step], feed_dict=feed_dict)
        # print('cnn_out:', c_out[0:3, ...])
        print('logits', logits[0:3, ...])
        print('预测值:', pre)
        print('loss:', xen)
        print()

    sess.close()


if __name__ == '__main__':
    main()


