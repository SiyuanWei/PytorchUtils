import math

import cv2
import numpy as np
from scipy.ndimage.interpolation import rotate, shift, zoom

SCIPY_INETRPOLATIONs = {'bilinear': 3, 'nearest': 0}
CV2_INTERPOLATIOONs = {'bilinear': cv2.INTER_LINEAR, 'nearest': cv2.INTER_NEAREST}
CAXIS = 2  # channel通道的axis

# 参考pytorch中_get_inverse_affine_matrix函数
def get_affine_matrix(img_size, degree, shift, scale, shear):
    center = (img.shape[0] * 0.5 + 0.5, img.shape[1] * 0.5 + 0.5)
    angle = math.radians(degree)
    shear = math.radians(shear)
    scale = scale

    # 放缩变化和剪切变化
    d = math.cos(angle + shear) * math.cos(angle) + math.sin(angle + shear) * math.sin(angle)
    matrix = [
        math.cos(angle + shear), math.sin(angle + shear), 0,
        -math.sin(angle), math.cos(angle), 0
    ]
    matrix = [scale / d * m for m in matrix]

    # # 平移变换
    matrix[2] += matrix[0] * (-center[0] - shift[0]) + matrix[1] * (-center[1] - shift[1])
    matrix[5] += matrix[3] * (-center[0] - shift[0]) + matrix[4] * (-center[1] - shift[1])

    # Apply center translation: C * RSS^-1 * C^-1 * T^-1
    matrix[2] += center[0]
    matrix[5] += center[1]
    matrix = np.array(matrix).reshape((2, 3))
    return matrix


def affine2d_transform(image: np.ndarray, degree: (int, float), shifts: (int, float, tuple),
                       scales: (int, float, tuple), shear: (int, float), interp):
    if isinstance(scales, (int, float)):
        scales = (scales, scales)
    else:
        assert len(scales) == 2
    if isinstance(shifts, (int, float)):
        shifts = (shifts, shifts)
    else:
        assert len(shifts) == 2

    shape = image.shape[0:2]
    shifts = tuple(sh * s for sh, s in zip(shifts, shape))
    # 参考pytorch中的函数 生成一个仿射矩阵， 包含了四种变换
    # opencv和scipy里自带的affine transform只能表示旋转 位移 缩放三种变换，没有剪切变换
    matrix = get_affine_matrix(image.shape, degree, shifts, scales, shear)
    # skimage scipy PIL 读取的图片是H*W ，opencv是W*H
    return cv2.warpAffine(image, matrix, (shape[1], shape[0]),
                          flags=CV2_INTERPOLATIOONs[interp])


if __name__ == '__main__':
    import skimage.io as io

    img = io.imread(r'D:\dataset\DSB\image\0002.png')
    print(img.shape, img.dtype, type(img))
    io.imshow(img)
    io.show()
    import time

    s1 = time.time()
    img = np.flip(img, 0)
    # img = affine2d_transform(img, 0, (0, 0), 1, 15, 'nearest')
    print(time.time() - s1, img.shape, img.dtype)
    io.imshow(img)
    io.show()
