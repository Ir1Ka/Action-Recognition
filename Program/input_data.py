#! /usr/bin/python3
# coding:utf-8


"""
数据输入和预处理
"""

import numpy as np
import cv2
import os
import random
from queue import Queue
from threading import Thread
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor
from math import ceil

import tools


def read_video(fname, sample_rate=1, dtype=None,
               is_test=False, target_resolution=None):
    '''
    读取指定文件，返回一个numpy类型的4维张量,元素类型为np.uint8，和视频帧率。
    同时进行下采样和提取光流信息
    '''
    if isinstance(fname, str) is False:
        raise TypeError('''文件名参数类型错误''')
    if isinstance(sample_rate, int) is False or sample_rate <= 0:
        raise TypeError('''视频下采样率必须是整数且必须大于零''')

    cap = cv2.VideoCapture(fname)
    if cap.isOpened() is False:
        raise Exception('''文件打开失败''')

    frames = []
    # 获取帧率
    # fps = cap.get(cv2.CV_CAP_PROP_FPS)
    fps = 0
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if i % sample_rate == 0:
            if is_test:
                target_h, target_w = target_resolution
                frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
            frame.shape = (1,) + frame.shape
            frames.append(frame)
        i += 1
    cap.release()
    video = np.concatenate(frames)
    if dtype is not None:
        video = video.astype(dtype)
    return video, fps


def read_videos(fname_list, sample_rate=1, dtype=None,
                is_test=False, target_resolution=None):
    '''
    读取一批视频，返回按传入文件名是顺序的视频list
    '''
    video_list = []
    for fname in fname_list:
        video, _ = read_video(
            fname, sample_rate=sample_rate, dtype=dtype,
            is_test=is_test, target_resolution=target_resolution
        )
        video_list.append(video)

    return video_list


def down_sampling(inputs, axis=0, sample_rate=1):
    '''
    对输入的张量按指定轴进行下采样。

    Args:

        inputs:输入张量,np.ndArray对象。

        axis:指定下采样轴。

        sample_rate:采样步长。

    Returns:

        out:采样后的张量。
    '''
    if isinstance(inputs, np.ndarray) is False:
        raise TypeError('''输入变量类别错误''')
    if isinstance(sample_rate, int) is False or sample_rate <= 0:
        raise TypeError('''视频下采样率必须是整数且必须大于零''')

    num_axis = len(inputs.shape)
    if axis >= num_axis:
        raise ValueError('''指定轴不存在''')

    slc = [slice(None)] * num_axis
    slc[axis] = slice(None, None, sample_rate)
    return inputs[slc]


def walk_fname_cls(rootdir, extensions=['.avi']):
    '''
    遍历指定目录下的所有的指定后缀名的文件，并使用list返回遍历到的所有文件名。

    args:

        rootdir:要遍历的根文件夹，最好是提供全路径。

        extensions:后缀名列表，即文件名的后面几个字符的匹配，可以是'.avi'的形式，也可以是't.avi'的形式。

    returns:

        total_fname_list:遍历出的符合后缀条件的文件的全路径列表。
    '''
    if isinstance(extensions, str) is True:
        extensions = [extensions]

    fname_to_cls = {}
    cls_set = {}

    # os.walk()方法会把目录看成一棵树，然后使用先序遍历算法遍历出所有的中间结点（目录）和叶子结点（文件或空目录）
    for root, _, fnames in os.walk(rootdir):
        for fname in fnames:
            for extension in extensions:
                if fname.endswith(extension) is True:
                    _, cls_ = os.path.split(root)
                    fname = os.path.join(root, fname)
                    fname_to_cls[fname] = cls_
                    cls_set[cls_] = None
                    break

    if len(fname_to_cls) < 1:
        raise ValueError('''数据为空''')
    cls_list = list(cls_set.keys())
    cls_list.sort()
    fname_to_label = {}
    for fname, cls_ in fname_to_cls.items():
        fname_to_label[fname] = cls_list.index(cls_)

    return fname_to_label, cls_list


class batchGenerator(object):
    def __init__(self):
        self.extensions = ['.avi']
        # 训练集参数
        self.train_dir = 'data/train_data'
        self.train_fname_to_label, self.cls_list = walk_fname_cls(
            self.train_dir, extensions=self.extensions
        )
        self.train_fname_list = list(self.train_fname_to_label.keys())
        random.shuffle(self.train_fname_list)
        self.train_batchs = Queue(maxsize=2)

        self.data_num = len(self.train_fname_list)
        self.epochs = 1000
        self.epochs_count = 0
        self.start = 0

        # 验证集参数
        self.test_dir = 'data/test_data'
        self.test_fname_to_label, _cls_list = walk_fname_cls(
            self.test_dir, extensions=self.extensions
        )
        if self.cls_list != _cls_list:
            raise ValueError('''测试集类别和训练集类别不同''')
        self.test_fname_list = list(self.test_fname_to_label.keys())
        self.test_num = len(self.test_fname_list)
        self.test_batchs = Queue(maxsize=2)

        # 进程池
        self.cpu_count = cpu_count()
        # 所有数据处理的共享线程池，1.5倍于cpu核数
        self.pool = ThreadPoolExecutor(max_workers=ceil(self.cpu_count * 1.5))

        # 一些超参数
        self.batch_size = 16
        self.test_size_extend = 1  # 测试时 batch 的扩大倍数
        self.test_batch_size = min(
            [ceil(self.batch_size * self.test_size_extend), ceil(self.test_num//2)]
        )
        self.min_frame_len = 60
        self.delta = 20
        self.max_frame_len = self.min_frame_len + self.delta
        self.frame_step = 2
        self.frame_sample_rate = 3
        self.height, self.width, self.channels = 96, 128, 3

    def random_frame_len(self):
        var0 = random.randint(0, ceil(self.delta/self.frame_step))
        target_frame_len = self.min_frame_len + self.frame_step*var0
        return min([target_frame_len, self.max_frame_len])

    def data_augment(self, video):
        # 随机翻转
        video = tools.random_flip(video, axis=[-2, -3])
        video = tools.random_crop_video(
            video, target_resolution=[self.height, self.width]
        )
        return video

    def repeat(self, video, target_min_len, use_random=False, max_random_rate=0.05):
        length = video.shape[0]
        if target_min_len <= length:
            return video

        new_length = length
        fragment_list = [video]
        while new_length <= target_min_len:
            if use_random:
                random_len = random.randint(0, round(length*max_random_rate))
                random_frames = np.random.randint(
                    0, 255, (random_len,)+video.shape[1:]
                ).astype(video.dtype)
                fragment_list.append(random_frames)
                new_length += random_len
            fragment_list.append(video)
            new_length += length
        return np.concatenate(fragment_list)

    def normalize(self, video):
        # 减去均值，除以方差
        channels = video.shape[-1]
        video_splits = np.split(video, channels, axis=-1)
        v = []
        for video_split in video_splits:
            mean = np.mean(video_split)
            stddev = np.std(video_split)
            video_split = (video_split-mean)/stddev
            # video_split.shape = video_split.shape + (1,)
            v.append(video_split)
        return np.concatenate(v, axis=-1)

    def handle_data(self, fname_list, target_frame_len, dtype=None, is_test=False):
        video_src_list = read_videos(
            fname_list,
            sample_rate=self.frame_sample_rate,
            dtype=dtype,
            is_test=is_test,
            target_resolution=(self.height, self.width)  # (高， 宽)
        )

        video_list = []
        labels = []  # 稀疏标签
        for index in range(len(fname_list)):
            try:
                if is_test:
                    label = self.test_fname_to_label[fname_list[index]]
                else:
                    label = self.train_fname_to_label[fname_list[index]]
            except:
                # 用于最终 demo 时，可能数据不会出现在测试集和训练集中
                label = None
            labels.append(label)
            video_src = video_src_list[index]
            if not is_test:
                video_src = self.data_augment(video_src)
            video_src = self.repeat(video_src, target_frame_len)
            video_len = video_src.shape[0]
            if video_len > target_frame_len:
                if is_test:
                    offset = 0
                else:
                    step_len = target_frame_len // 4
                    delta = video_len - target_frame_len
                    max_random_int = ceil(delta/step_len)
                    random_int = random.randint(0, max_random_int)
                    if random_int < max_random_int:
                        offset = random_int * step_len
                    else:
                        offset = delta
                video_src = video_src[offset: offset+target_frame_len]
            video = self.normalize(video_src)
            video.shape = (1,) + video.shape
            video_list.append(video)
        return video_list, labels

    def handle_batch(self, batch_fname, is_test=False):
        # 随机一个视频长度
        random_frame_len = self.random_frame_len()

        fname_count = len(batch_fname)
        consult = fname_count // self.cpu_count
        remainder = fname_count % self.cpu_count

        if consult == 0:
            handle_threads_size = remainder
        else:
            handle_threads_size = self.cpu_count

        futures = []
        start = 0
        for i in range(handle_threads_size):
            current_fname_count = consult
            if i < remainder:
                current_fname_count += 1
            elif current_fname_count == 0:
                break
            future = self.pool.submit(
                self.handle_data,
                batch_fname[start: start+current_fname_count],
                random_frame_len,
                dtype=np.float32,
                is_test=is_test
            )
            futures.append(future)
            start += current_fname_count
        video_list = []
        labels = []
        for future in futures:
            v_list, label = future.result()
            video_list.extend(v_list)
            labels.extend(label)
        return np.concatenate(video_list), labels

    def read_train_batch(self):
        """
        读取一个batch的视频
        """
        # 判断是否遍历完一次训练集，丢弃最后的不足一个 batch 的数据
        if self.start+self.batch_size > self.data_num:
            random.shuffle(self.train_fname_list)
            self.start = 0
            self.epochs_count += 1
        if self.epochs_count >= self.epochs:
            # 表示训练结束
            return None, None

        fname_list = self.train_fname_list[self.start: self.batch_size+self.start]
        self.start += self.batch_size  # 更新下一次的起始读取位置
        return self.handle_batch(fname_list)

    def read_test_batch(self):
        random.shuffle(self.test_fname_list)
        start = 0
        test_num = len(self.test_fname_list)

        while True:
            if start >= test_num:
                break
            # 读文件
            fname_list = self.test_fname_list[start: start+self.test_batch_size]
            start += self.test_batch_size  # 更新下一次的起始读取位置
            yield self.handle_batch(fname_list, is_test=True)

    def test_batch_generator(self):
        test_batch_reader = self.read_test_batch()
        try:
            for batch_labels in test_batch_reader:
                self.test_batchs.put(batch_labels)
        finally:
            self.test_batchs.put((None, None))


class trainBatchGeneratorThread(Thread):
    def __init__(self, batch_generator):
        Thread.__init__(self)
        self.batch_generator = batch_generator

    def run(self):
        self.batch_generator.epochs_count = 0
        while True:
            batch_labels = self.batch_generator.read_train_batch()
            self.batch_generator.train_batchs.put(batch_labels)
            if batch_labels[0] is None:
                break

    def get_train_batch(self, timeout=10):
        return self.batch_generator.train_batchs.get(timeout=timeout)








