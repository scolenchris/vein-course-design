# model/lbp.py
import cv2
import numpy as np
from skimage.feature import local_binary_pattern


class LBPFeatureExtractor:
    def __init__(
        self, radius=1, n_points=8, grid_x=8, grid_y=8, target_size=(256, 256)
    ):
        """
        初始化 LBP 特征提取器
        """
        self.radius = radius
        self.n_points = n_points
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.target_size = target_size
        self.n_bins = int(n_points * (n_points - 1) + 3)

    def preprocess(self, image_path):
        """读取图像，转灰度，并 Resize 到统一尺寸"""
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"无法读取图像: {image_path}")

        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        gray_eq = cv2.equalizeHist(gray)
        gray_resized = cv2.resize(
            gray_eq, self.target_size, interpolation=cv2.INTER_CUBIC
        )
        return gray_resized

    def get_lbp_map(self, image):
        """获取 Uniform LBP 编码图"""
        return local_binary_pattern(image, self.n_points, self.radius, method="uniform")

    def extract_features(self, image_path, return_lbp_map=False):
        """
        核心流程：预处理 -> LBP编码 -> 分块 -> 直方图统计 -> 归一化 -> 串接
        """
        img = self.preprocess(image_path)
        lbp_map = self.get_lbp_map(img)

        h, w = lbp_map.shape
        dy, dx = int(h / self.grid_y), int(w / self.grid_x)
        histograms = []
        max_bins = int(lbp_map.max() + 1)

        for r in range(self.grid_y):
            for c in range(self.grid_x):
                block = lbp_map[r * dy : (r + 1) * dy, c * dx : (c + 1) * dx]
                hist, _ = np.histogram(
                    block.ravel(), bins=max_bins, range=(0, max_bins)
                )

                # L1 归一化
                hist = hist.astype("float")
                hist /= hist.sum() + 1e-7
                histograms.append(hist)

        final_feature = np.concatenate(histograms)

        if return_lbp_map:
            return final_feature, img, lbp_map
        return final_feature

    @staticmethod
    def match_histograms(feat1, feat2):
        """
        直方图交叉 (Histogram Intersection)
        """
        if feat1.shape != feat2.shape:
            raise ValueError(f"特征向量维度不一致: {feat1.shape} vs {feat2.shape}")

        min_vals = np.minimum(feat1, feat2)
        score = np.sum(min_vals)
        return score
