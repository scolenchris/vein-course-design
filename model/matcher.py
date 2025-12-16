# model/matcher.py
import cv2
import numpy as np


def read_binary_image(image_path, size=(500, 500)):
    """
    读取图像并转换为二值纹理矩阵 (0和1)。
    为了保证能够进行像素级比对，这里默认将图像resize到统一大小。
    """
    # 以灰度模式读取
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")

    # 调整大小以确保两张图可以进行像素点运算
    img = cv2.resize(img, size)

    # 二值化处理 (假设输入已经是二值图，但为了保险进行一次阈值处理)
    # 将像素值 > 127 的置为 1，其余为 0
    _, binary_img = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)

    return binary_img.astype(np.int8)


def calculate_iou_score(img1, img2):
    """
    计算两张二值纹理前景图像的匹配分数。
    公式：交集像素点数 / 并集像素点数 (IoU)
    """
    if img1.shape != img2.shape:
        # 如果尺寸不一致，强制resize img2 匹配 img1
        img2 = cv2.resize(
            img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_NEAREST
        )

    # 计算交集 (逻辑与)
    intersection = np.logical_and(img1, img2)
    # 计算并集 (逻辑或)
    union = np.logical_or(img1, img2)

    intersection_count = np.sum(intersection)
    union_count = np.sum(union)

    if union_count == 0:
        return 0.0

    score = intersection_count / union_count
    return score
