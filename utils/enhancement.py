import cv2
import numpy as np


def gray_normalization(img):
    """
    灰度归一化：将像素值线性拉伸到 0-255
    """
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def global_histogram_equalization(img):
    """
    全局直方图均衡化 (HE)
    """
    return cv2.equalizeHist(img)


def clahe_enhancement(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    限制对比度自适应直方图均衡化 (CLAHE)
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(img)


def single_scale_retinex(img, sigma):
    """
    单尺度 Retinex (SSR)
    """
    # 转换为浮点数进行计算
    img_f = img.astype(float) + 1.0
    # 高斯模糊
    img_gaussian = cv2.GaussianBlur(img, (0, 0), sigma) + 1.0
    # Log 域相减
    retinex = np.log(img_f) - np.log(img_gaussian)
    return retinex


def multi_scale_retinex(img, sigma_list=[15, 80, 250]):
    """
    多尺度 Retinex (MSR)
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    retinex = np.zeros_like(img, dtype=float)
    for sigma in sigma_list:
        retinex += single_scale_retinex(img, sigma)

    retinex = retinex / len(sigma_list)

    # 将 Log 域结果映射回 0-255
    # 这里使用简单的均值方差归一化或 MinMax
    retinex = cv2.normalize(retinex, None, 0, 255, cv2.NORM_MINMAX)
    return retinex.astype(np.uint8)


def apply_enhancements(img, config):
    """
    根据配置按顺序应用增强
    建议顺序：Retinex -> HE/CLAHE -> Normalization
    """
    res_img = img.copy()

    # 1. Retinex (通常先做，因为它处理的是光照)
    if config.get("enable_retinex", False):
        res_img = multi_scale_retinex(res_img)

    # 2. 直方图均衡化 (全局)
    if config.get("enable_he", False):
        res_img = global_histogram_equalization(res_img)

    # 3. CLAHE (局部)
    if config.get("enable_clahe", False):
        res_img = clahe_enhancement(res_img)

    # 4. 归一化 (最后确保范围正确)
    if config.get("enable_norm", False):
        res_img = gray_normalization(res_img)

    return res_img
