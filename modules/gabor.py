import cv2
import numpy as np
import random


def _apply_clahe(img, clip_limit=2.0, grid_size=(8, 8)):
    """(内部函数) CLAHE 自适应直方图均衡化"""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return clahe.apply(gray)
    return clahe.apply(img)


def post_process_binary(img_float_or_u8):
    """
    (内部函数) 统一后处理：归一化 -> Otsu二值化 -> 形态学去噪
    """
    # 1. 归一化到 0-255
    if img_float_or_u8.dtype != np.uint8:
        norm_img = cv2.normalize(img_float_or_u8, None, 0, 255, cv2.NORM_MINMAX).astype(
            np.uint8
        )
    else:
        norm_img = img_float_or_u8

    # 2. Otsu 二值化
    # 自动计算阈值
    ret, binary_img = cv2.threshold(
        norm_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    ret, binary_img = cv2.threshold(norm_img, ret * 0.8, 255, cv2.THRESH_BINARY)
    # 经验值：有时候 Otsu 阈值偏低，稍微提高一点可以减少噪声
    # 这里保持标准 Otsu，如果噪声多，可以取消下面这行的注释
    # ret, binary_img = cv2.threshold(norm_img, ret * 1.1, 255, cv2.THRESH_BINARY)

    # 3. 形态学去噪 (开运算去除小白点)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)
    # 返回反色二值化图像
    binary_img = cv2.bitwise_not(binary_img)
    return binary_img


def _build_gabor_bank(ksize_list=[7, 9], directions=8, lamda=9, sigma_ratio=5.5):
    """
    构造 Gabor 滤波器组
    :param ksize_list: 卷积核尺寸列表
    :param directions: 方向数量 (例如 8 代表每 22.5 度一个)
    :param lamda: 波长
    :param sigma_ratio: sigma 与 ksize 的比例 (用于控制高斯包络)
    """
    filters = []
    for ksize in ksize_list:
        sigma = ksize * sigma_ratio  # 动态 sigma，或直接固定如 4.0
        for theta in np.arange(0, np.pi, np.pi / directions):
            # params: (ksize, ksize), sigma, theta, lamda, gamma, psi
            # gamma=0.5 (空间纵横比), psi=0 (相位偏移)
            kern = cv2.getGaborKernel(
                (ksize, ksize), sigma, theta, lamda, 0.5, 0, ktype=cv2.CV_32F
            )
            # 归一化，保持响应幅度一致
            kern /= 1.5 * kern.sum()
            filters.append(kern)
    return filters


def gabor_vein_extract(raw_img, ksizes=[7, 9, 11], directions=8):
    """
    Gabor 滤波静脉提取算法接口
    优化了原代码的 Python 循环，使用矩阵运算加速

    :param raw_img: 输入图像
    :param ksizes: 使用的 Gabor 核大小列表 (多尺度融合)
    :param directions: 方向数量
    :return: 二值化静脉图像
    """
    # 1. 预处理
    if len(raw_img.shape) == 3:
        gray = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = raw_img.copy()

    # 使用 CLAHE 增强局部对比度，这对 Gabor 响应很重要
    gray = _apply_clahe(gray)

    # 2. 构建滤波器组
    filters = _build_gabor_bank(ksize_list=ksizes, directions=directions)

    # 3. 滤波与融合 (Max Response)
    # 使用 float32 累加器防止溢出
    accum_response = np.zeros_like(gray, dtype=np.float32)

    for kern in filters:
        # cv2.filter2D 极其高效
        fimg = cv2.filter2D(gray, cv2.CV_32F, kern)
        # 取每个像素在所有滤波器下的最大响应
        np.maximum(accum_response, fimg, accum_response)

    # 4. 后处理与二值化
    # 将响应图转回 uint8 范围
    # 注意：Gabor 响应可能有负值（尽管我们只取了 max，通常是正的），先 clip 再 normalize
    accum_response = np.clip(accum_response, 0, None)

    # 复用标准的二值化处理流程
    # return accum_response
    return post_process_binary(accum_response)
