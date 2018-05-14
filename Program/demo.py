#! /usr/bin/python3
# coding:utf-8
import tensorflow as tf
import numpy as np
import sys

from train import trainer
from tools import timer


def recog(fname_or_list, train=None):
    """
    识别 demo，输入一个或多个视频文件路径，返回识别结果
    """
    if train is None:
        train = trainer(is_recog=True)
        train.saver.restore(train.sess, train.model_fname)
    if not isinstance(fname_or_list, list):
        fname_or_list = [fname_or_list]
    with timer('识别视频') as t:
        probs, labels = train.recog(fname_or_list)
        delta = t.duration()
    is_all_test_set = False
    if fname_or_list == train.batch_generator.test_fname_list:
        is_all_test_set = True
        top_accs = [0 for _ in range(5)]
    for i in range(len(probs)):
        print('--------------------------------------------------')
        fname = fname_or_list[i]
        prob = probs[i]
        label = labels[i]
        cls_name = train.batch_generator.cls_list[label]
        str_format = '%s 的预测标签为：%d，预测类别为：%s' % (fname, label, cls_name)
        print(str_format)
        prob = prob.tolist()
        if is_all_test_set:
            true_label = train.batch_generator.test_fname_to_label[fname]
        print('Top-5 预测概率值分别为：')
        top_5_index = np.argsort(prob)[::-1][:5]
        for i in range(0, len(top_5_index), 2):
            index = top_5_index[i]
            if is_all_test_set:
                if true_label in top_5_index[:i+1]:
                    top_accs[i] += 1
            str_format = '标签：%d - 类别名：%s - 概率值：%f'
            cls_name = train.batch_generator.cls_list[index]
            p = prob[index]
            print(str_format % (index, cls_name, p))

    print('--------------------------------------------------')
    print('以上为所有输入视频的预测标签、类别名和预测概率值')
    print('总视频数为：%d' % len(fname_or_list))
    print('视频总用时：%.3f' % delta)
    speed = delta / len(fname_or_list)
    print('识别平均速度为：%.3f sec/视频' % speed)

    if is_all_test_set:
        print('++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('当前使用所有测试集作为演示，可计算准确率，准确率如下：')
        for i in range(0, len(top_accs), 2):
            print('Top-%d准确率为：%.4f' % (i+1, top_accs[i] / len(fname_or_list)))
    train.close()

if __name__ == '__main__':
    fname_or_list = sys.argv[1:]
    with timer('恢复模型和参数'):
        train = trainer(is_recog=True)
        if len(fname_or_list) == 0:
            print('没有指定视频，使用所有测试集作为演示')
            fname_or_list = train.batch_generator.test_fname_list
        train.saver.restore(train.sess, train.model_fname)
    recog(fname_or_list, train=train)

