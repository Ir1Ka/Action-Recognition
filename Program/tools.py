# coding:utf-8

import time
import numpy as np
import random
import tensorflow as tf
import cv2
from threading import current_thread


class timer(object):
    """
    计时器
    """

    def __init__(self, hint):
        self.start = time.time()
        self.hint = hint
        print(self.hint, ' <-- started.')

    def __enter__(self):
        """
        with语句返回本身
        """
        return self

    def __exit__(self, type, value, trace):
        print(self.hint, ' --> finished. 历时：%.3f sec' % (time.time()-self.start))

    def duration(self):
        """
        从with语句开始到调用时刻的历时，返回以秒为单位的时间
        """
        return time.time() - self.start


def flip(x, axis=0):
    """
    按axis指定的轴翻转，轴可正索引，可负索引。超出范围将忽略
    """
    if isinstance(axis, int):
        axis = [axis]
    axis = [ax if 0 <= ax else ax+x.ndim for ax in axis]
    return x[[slice(-1, -1-x.shape[dim], -1) if dim in axis else slice(None, None, None) for dim in range(x.ndim)]]


def random_flip(x, axis=0):
    """
    按指定轴随机翻转
    """
    if isinstance(axis, int):
        axis = [axis]
    axis = [ax if 0 == random.randint(0, 1) else x.ndim for ax in axis]
    return flip(x, axis=axis)


def draw_flow(img, flow, noise=2, step=16, draw_location_point=False):
    """
    绘制光流图

    Args:

        flow:光流信息
    """
    if noise < 0.:
        noise = 0.
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)

    lines = lines.tolist()
    i = 0
    lines_num = len(lines)
    while i < lines_num:
        (x1, y1), (x2, y2) = lines[i]
        if abs((x1**2 + y1**2)**0.5 - (x2**2 + y2**2)**0.5) <= noise:
            del lines[i]
            lines_num -= 1
        else:
            i += 1
    lines = np.array(lines)

    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))

    if draw_location_point:
        # 绘制定位点
        for (x1, y1), (x2, y2) in lines:
            cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)

    return vis


prevgrays = {}


def clean_prevgray(id=None):
    """
    清空指定`id`的前一帧灰度图像
    """
    global prevgrays
    if id is None:
        id = current_thread().getName()
    if id in prevgrays.keys():
        del prevgrays[id]


def get_flow_with_farneback(img, id=None, noise=2.):
    """
    使用Farneback算法得到光流图，图像为灰色三通道，光流信息为绿色，有利于突出光流信息

    Args:

        img:彩色图像

        id:标志，用于多线程重入，默认值时获取当前线程名作为ID，也可自己指定

    Returns:

        out:绘制的光流图像
    """
    global prevgrays
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if id is None:
        id = current_thread().getName()
    if id in prevgrays.keys():
        prevgray = prevgrays[id]
    else:
        prevgrays[id] = gray
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    flow = cv2.calcOpticalFlowFarneback(
        prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    prevgrays[id] = gray

    return draw_flow(gray, flow, noise=noise)


def draw_video(video):
    imgs = []
    for img in video:
        img = get_flow_with_farneback(img)
        img.shape = (1,) + img.shape
        imgs.append(img)
    clean_prevgray()
    return np.concatenate(imgs, axis=0)


def rotate_image(img, angle, crop=True):
    '''
    定义旋转函数：
    angle是逆时针旋转的角度
    crop是个布尔值，表明是否要裁剪去除黑边
    '''
    h, w = img.shape[:2]
    # 旋转角度的周期是360°
    angle %= 360
    # 用OpenCV内置函数计算仿射矩阵
    M_rotate = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
    # 得到旋转后的图像
    img_rotated = cv2.warpAffine(img, M_rotate, (w, h))

    # 如果需要裁剪去除黑边
    if crop:
        # 对于裁剪角度的等效周期是180°
        angle_crop = angle % 180
        # 并且关于90°对称
        if angle_crop > 90:
            angle_crop = 180 - angle_crop
        # 转化角度为弧度
        theta = angle_crop * np.pi / 180.0
        # 计算高宽比
        hw_ratio = float(h) / float(w)
        # 计算裁剪边长系数的分子项
        tan_theta = np.tan(theta)
        numerator = np.cos(theta) + np.sin(theta) * tan_theta
        # 计算分母项中和宽高比相关的项
        r = hw_ratio if h > w else 1 / hw_ratio
        # 计算分母项
        denominator = r * tan_theta + 1
        # 计算最终的边长系数
        crop_mult = numerator / denominator
        # 得到裁剪区域
        w_crop = int(round(crop_mult*w))
        h_crop = int(round(crop_mult*h))
        x0 = int((w-w_crop)/2)
        y0 = int((h-h_crop)/2)

        img_rotated = crop_image(img_rotated, x0, y0, w_crop, h_crop)

    return img_rotated


def random_rotate(img, angle_vari, p_crop):
    '''
    随机旋转
    angle_vari是旋转角度的范围[-angle_vari, angle_vari)
    p_crop是要进行去黑边裁剪的比例
    '''
    angle = np.random.uniform(-angle_vari, angle_vari)
    crop = False if np.random.random() > p_crop else True
    return rotate_image(img, angle, crop)


def crop_video(video, crop_resolution, origin=[0, 0], target_resolution=None):
    if target_resolution is None:
        target_resolution = video.shape[-3:-1]
    src_h, src_w = video.shape[-3:-1]
    crop_y, crop_x = crop_resolution
    if origin[0]+crop_y > src_h or origin[1]+crop_x > src_w:
        raise ValueError('''裁剪到了边缘''')
    crop_video = video[
        ..., origin[0]: origin[0]+crop_y, origin[1]: origin[1]+crop_x, :
    ]
    new_fragment_list = []
    target_h, target_w = target_resolution
    for im in video:
        im = cv2.resize(im, (target_w, target_h), interpolation=cv2.INTER_AREA)
        im.shape = (1,) + im.shape
        new_fragment_list.append(im)
    return np.concatenate(new_fragment_list)


def random_crop_video(video, min_crop_rate=4/5, target_resolution=None):
    crop_rate = random.uniform(min_crop_rate, 1.)
    src_h, src_w = video.shape[-3:-1]
    crop_h, crop_w = round(src_h*crop_rate), round(src_w*crop_rate)
    origin = [random.randint(0, src_h-crop_h), random.randint(0, src_w-crop_w)]
    return crop_video(video, [crop_h, crop_w], origin=origin, target_resolution=target_resolution)


layers_out = []


def add_data(data):
    global layers_out
    layers_out.append(data)


def get_layers_out():
    global layers_out
    return layers_out


def softmax(x, axis=-1):
    """
    手动softmax
    """
    exp = np.exp(x)
    sum_ = np.sum(exp, axis=axis)
    sum_.shape = sum_.shape + (1,)
    return exp / sum_


