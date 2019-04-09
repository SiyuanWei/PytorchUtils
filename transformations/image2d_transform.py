import random

import cv2
import numpy as np
from skimage.util import random_noise

from transformations.image2d_func import affine2d_transform


class ReshapeHWC(object):
    def __init__(self, image_reshape: bool, label_reshape: bool):
        self.image_reshape, self.label_reshape = image_reshape, label_reshape

    def __call__(self, image: np.ndarray, label: np.ndarray):
        if self.image_reshape and len(image.shape) == 2:
            image = image[:, np.newaxis]
        if self.label_reshape:
            label = label[:, np.newaxis]
        return image, label


class Resize(object):
    def __init__(self, out_size: (int, float, tuple)):
        if isinstance(out_size, (int, float)):
            self.out_size = (out_size, out_size)
        else:
            assert len(out_size) == 2
            self.out_size = out_size

    def __call__(self, image: np.ndarray, label: np.ndarray):
        new_h, new_w = self.out_size
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        return image, label


class Rescale(object):
    def __init__(self, scales: (int, float)):
        self.scales = (float(scales), float(scales))

    def __call__(self, image: np.ndarray, label: np.ndarray):
        shape = image.shape[0:2]
        new_h, new_w = self.scales[0] * shape[0], self.scales[1] * shape[1]
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        return image, label


class RandomScale(object):
    def __init__(self, max: float, min: float, step: float):
        assert max >= min
        assert step <= 1
        self.sample_list = np.arange(min, max + step, step).tolist()

    def __call__(self, image: np.ndarray, label: np.ndarray):
        scale = random.sample(self.sample_list, 1)[0]  # random sample 返回一个list
        shape = image.shape[0:2]
        new_h, new_w = scale * shape[0], scale * shape[1]
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        return image, label


class RandomAffine(object):
    def __init__(self, degrees: (tuple), shifts: (tuple),
                 scales: (tuple), shears: (tuple)):
        self.degrees, self.shifts, self.scales, self.shears = degrees, shifts, scales, shears

    @staticmethod
    def _random_params(degrees, shifts, scales, shears):
        degree = random.uniform(degrees[0], degrees[1])
        shift = random.uniform(shifts[0], shifts[1])
        scale = random.uniform(scales[0], scales[1])
        shear = random.uniform(shears[0], shears[1])
        return degree, shift, scale, shear

    def __call__(self, image: np.ndarray, label: np.ndarray):
        degree, shift, scale, shear = self._random_params(self.degrees, self.shifts, self.scales, self.shears)
        image = affine2d_transform(image, degree, shift, scale, shear, 'bilinear')
        label = affine2d_transform(label, degree, shift, scale, shear, 'nearest')
        return image, label


class RandomFlip(object):
    def __init__(self, h: bool, v: bool, prob: float):
        # h：水平翻转，v：垂直翻转
        assert prob <= 0.5 and prob >= 0
        self.h, self.v, self.prob = h, v, prob

    def __call__(self, image: np.ndarray, label: np.ndarray):

        if self.h and random.random() <= self.prob:
            image = np.flip(image, 1)
            label = np.flip(label, 1)
        if self.v and random.random() <= self.prob:
            image = np.flip(image, 0)
            label = np.flip(label, 0)

        return image, label


class ROIRandomCrop(object):
    def __init__(self, output_size: (int, tuple)):
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, image: np.ndarray, label: np.ndarray):

        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        assert (new_h <= h) and (new_w <= w)

        roi_indexs = np.nonzero(label)
        min_x, max_x = min(roi_indexs[0]), max(roi_indexs[0])
        min_y, max_y = min(roi_indexs[1]), max(roi_indexs[1])

        (x_low, x_high) = (min_x, max_x - new_h) \
            if max_x - min_x >= new_h else (max(max_x - new_h, 0), min(min_x, h - new_h))
        (y_low, y_high) = (min_y, max_y - new_w) \
            if max_y - min_y >= new_w else (max(max_y - new_w, 0), min(min_y, w - new_w))

        top = np.random.randint(x_low, x_high) if x_high > x_low else x_low
        left = np.random.randint(y_low, y_high) if y_high > y_low else y_low

        image = image[top: top + new_h,
                left: left + new_w]

        label = label[top: top + new_h,
                left: left + new_w]

        return image, label


class CenterCrop(object):
    def __init__(self, output_size: (int, tuple)):
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, image: np.ndarray, label: np.ndarray):

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        assert (new_h <= h) and (new_w <= w)

        top = (h - new_h) // 2
        left = (w - new_w) // 2

        image = image[top: top + new_h,
                left: left + new_w]

        label = label[top: top + new_h,
                left: left + new_w]

        return image, label


class RandomCrop(object):
    def __init__(self, output_size: (int, tuple)):
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, image: np.ndarray, label: np.ndarray):

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        assert (new_h <= h) and (new_w <= w)

        top = np.random.randint(0, h - new_h + 1)  # numpy的随机整数生成 上界比最大值小一
        left = np.random.randint(0, w - new_w + 1)

        image = image[top: top + new_h,
                left: left + new_w]

        label = label[top: top + new_h,
                left: left + new_w]

        return image, label


class RandomNoise(object):
    def __init__(self, mode: str):
        assert mode in ['gaussian', 'localvar', 'poisson', 'salt', 'pepper', 'speckle', 's&p']
        self.mode = mode

    def __call__(self, image: np.ndarray, label: np.ndarray):
        image = random_noise(image, mode=self.mode, clip=False)  # clip一定得是false 否则数据失真
        return image, label


class Normalize(object):
    def __init__(self, mean: tuple, std: tuple):
        assert len(mean) == len(std)
        self.mean = mean
        self.std = std

    def __call__(self, image: np.ndarray, label: np.ndarray):
        for i in range(image.shape[2]):
            image[:, :, i] = (image[:, :, i] - self.mean[i]) / self.std[i]
        return image, label
