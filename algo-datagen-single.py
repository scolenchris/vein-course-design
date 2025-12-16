import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# from modules.vein_algo import repeated_line_tracking
from modules.gabor import gabor_vein_extract
from modules.rlt import repeated_line_tracking
from modules.vein_algo import max_curvature_extract


def imfilter(src, kernel):
    """
    模拟 Matlab 的 imfilter 函数，使用 cv2.filter2D 实现。
    """
    # cv2.filter2D 默认就是相关操作 (correlation)，符合一般滤波需求
    # ddepth=-1 表示输出图像深度与原图相同
    return cv2.filter2D(src, -1, kernel)


def apply_clahe(img, clip_limit=2.0, grid_size=(8, 8)):
    """
    标准 CLAHE 增强
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return clahe.apply(gray)
    return clahe.apply(img)


def max_curvature(raw, mask=None, sigma=3.0, show=False):
    """
    最大曲率算法 (Maximum Curvature Method)

    :param raw: 原始输入图像 (BGR 或 灰度)
    :param mask: ROI 掩膜 (可选)
    :param sigma: 高斯核标准差，控制检测静脉的宽度 (默认 3.0，根据图片分辨率调整)
    :param show: 是否显示调试图
    :return: 二值化的静脉纹理图
    """
    print(f"Starting Maximum Curvature extraction (sigma={sigma})...")

    # 1. 预处理与增强
    if len(raw.shape) == 3:
        gray_raw = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    else:
        gray_raw = raw.copy()

    # 这里强制使用 CLAHE 进行增强，因为静脉通常对比度低
    gray = apply_clahe(gray_raw)

    if mask is None:
        mask = np.ones_like(gray)

    # 2. 构造高斯导数核
    win_size = int(np.ceil(4 * sigma))
    x = np.arange(-win_size, win_size + 1)
    y = np.arange(-win_size, win_size + 1)
    X, Y = np.meshgrid(x, y)

    # 基础高斯函数及其导数
    common_factor = 1 / (2 * np.pi * sigma**2)
    exp_part = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    h = common_factor * exp_part

    hx = (-X / (sigma**2)) * h
    hxx = ((X**2 - sigma**2) / (sigma**4)) * h
    hy = hx.T
    hyy = hxx.T
    hxy = ((X * Y) / (sigma**4)) * h

    # 3. 滤波计算梯度与曲率
    fx = imfilter(gray.astype(np.float32), hx)
    fxx = imfilter(gray.astype(np.float32), hxx)
    fy = imfilter(gray.astype(np.float32), hy)
    fyy = imfilter(gray.astype(np.float32), hyy)
    fxy = imfilter(gray.astype(np.float32), hxy)

    f1 = 0.5 * np.sqrt(2) * (fx + fy)
    f2 = 0.5 * np.sqrt(2) * (fx - fy)
    f11 = 0.5 * fxx + fxy + 0.5 * fyy
    f22 = 0.5 * fxx - fxy + 0.5 * fyy

    # 4. 计算四个方向的曲率 k
    h_img, w_img = gray.shape
    k = np.zeros((h_img, w_img, 4))

    # 加上 1e-6 防止除以零
    k[:, :, 0] = (fxx / ((1 + fx**2) ** (1.5) + 1e-6)) * mask
    k[:, :, 1] = (fyy / ((1 + fy**2) ** (1.5) + 1e-6)) * mask
    k[:, :, 2] = (f11 / ((1 + f1**2) ** (1.5) + 1e-6)) * mask
    k[:, :, 3] = (f22 / ((1 + f2**2) ** (1.5) + 1e-6)) * mask

    # 5. 剖面计分 (Score Calculation) - 这是最耗时的部分
    Vt = np.zeros_like(gray, dtype=np.float32)

    print("  Processing Horizontal direction...")
    # Horizontal
    bla = k[:, :, 0] > 0
    for y in range(h_img):
        Wr = 0
        for x in range(w_img):
            if bla[y, x]:
                Wr += 1
            if Wr > 0 and (x == (w_img - 1) or not bla[y, x]):
                pos_end = x if x == (w_img - 1) else x - 1
                pos_start = pos_end - Wr + 1

                # 寻找局部最大值位置
                region = k[y, pos_start : pos_end + 1, 0]
                if region.size > 0:
                    pos_max = pos_start + np.argmax(region)
                    Vt[y, pos_max] += k[y, pos_max, 0] * Wr
                Wr = 0

    print("  Processing Vertical direction...")
    # Vertical
    bla = k[:, :, 1] > 0
    for x in range(w_img):
        Wr = 0
        for y in range(h_img):
            if bla[y, x]:
                Wr += 1
            if Wr > 0 and (y == (h_img - 1) or not bla[y, x]):
                pos_end = y if y == (h_img - 1) else y - 1
                pos_start = pos_end - Wr + 1

                region = k[pos_start : pos_end + 1, x, 1]
                if region.size > 0:
                    pos_max = pos_start + np.argmax(region)
                    Vt[pos_max, x] += k[pos_max, x, 1] * Wr
                Wr = 0

    print("  Processing Diagonal directions (Slow)...")
    # Direction: \
    bla = k[:, :, 2] > 0
    # 优化：不使用 while 循环，改用对角线遍历的数学方法略复杂，保持原逻辑但增加保护
    # 为保持代码逻辑一致性，这里保留原逻辑，但请注意效率较低
    for start in range(0, w_img + h_img - 1):
        x = start if start < w_img else 0
        y = 0 if start < w_img else start - w_img + 1
        Wr = 0
        while x < w_img and y < h_img:
            if bla[y, x]:
                Wr += 1

            is_edge = y == h_img - 1 or x == w_img - 1
            if Wr > 0 and (is_edge or not bla[y, x]):
                pos_x_end = x if is_edge else x - 1
                pos_y_end = y if is_edge else y - 1
                pos_x_start = pos_x_end - Wr + 1
                pos_y_start = pos_y_end - Wr + 1

                # 获取对角线上的值
                length = pos_x_end - pos_x_start + 1
                # 构造索引
                ys = np.arange(pos_y_start, pos_y_start + length)
                xs = np.arange(pos_x_start, pos_x_start + length)

                region = k[ys, xs, 2]
                if region.size > 0:
                    temp = np.argmax(region)
                    Vt[pos_y_start + temp, pos_x_start + temp] += region[temp] * Wr
                Wr = 0

            x += 1
            y += 1

    # Direction: /
    bla = k[:, :, 3] > 0
    for start in range(0, w_img + h_img - 1):
        x = start if start < w_img else 0
        y = h_img - 1 if start < w_img else w_img + h_img - start - 1
        Wr = 0
        while x < w_img and y >= 0:
            if bla[y, x]:
                Wr += 1

            is_edge = y == 0 or x == w_img - 1
            if Wr > 0 and (is_edge or not bla[y, x]):
                pos_x_end = x if is_edge else x - 1
                pos_y_end = y if is_edge else y + 1
                pos_x_start = pos_x_end - Wr + 1
                pos_y_start = pos_y_end + Wr - 1  # y是递减的，所以start比end大

                length = pos_x_end - pos_x_start + 1
                # y 递减，x 递增
                ys = np.arange(pos_y_start, pos_y_start - length, -1)
                xs = np.arange(pos_x_start, pos_x_start + length)

                # 边界检查，防止索引越界
                valid_len = min(len(ys), len(xs))
                if valid_len > 0:
                    ys = ys[:valid_len]
                    xs = xs[:valid_len]
                    region = k[ys, xs, 3]

                    if region.size > 0:
                        temp = np.argmax(region)
                        Vt[ys[temp], xs[temp]] += region[temp] * Wr
                Wr = 0

            x += 1
            y -= 1

    # 6. 连接静脉中心 (Connection)
    print("  Connecting vein centres...")
    Cd = np.zeros((h_img, w_img, 4))

    # 填充 Cd，处理边界时忽略最外圈，避免越界
    # Vectorized check for neighbors is hard due to min/max logic, keeping loops for safety
    # 为了加速，这里只做简单的切片操作（简化版），原逻辑略显繁琐
    # 如果要严格保持原逻辑：
    pad_Vt = np.pad(Vt, ((2, 2), (2, 2)), mode="constant")
    # 映射回原坐标系有点麻烦，这里还是用循环吧，但范围要注意
    # 原代码: range(2, w-3) -> 意味着忽略了边界
    for x in range(2, w_img - 3):
        for y in range(2, h_img - 3):
            Cd[y, x, 0] = min(Vt[y, x + 1 : x + 3].max(), Vt[y, x - 2 : x].max())
            Cd[y, x, 1] = min(Vt[y + 1 : y + 3, x].max(), Vt[y - 2 : y, x].max())
            Cd[y, x, 2] = min(
                Vt[y - 2 : y, x - 2 : x].max(), Vt[y + 1 : y + 3, x + 1 : x + 3].max()
            )  # diag
            Cd[y, x, 3] = min(
                Vt[y + 1 : y + 3, x - 2 : x].max(), Vt[y - 2 : y, x + 1 : x + 3].max()
            )  # anti-diag

    # 7. 合并与二值化
    img_veins = Cd.max(axis=2)

    # 简单的固定阈值二值化，通常这里可以改用 Otsu
    # 为了得到纯净的二值图，这里只要大于0就算 (原逻辑)，或者设一个小阈值
    _, img_veins_bin = cv2.threshold(img_veins, 0.01, 255, cv2.THRESH_BINARY)
    img_veins_bin = img_veins_bin.astype(np.uint8)

    if show:
        plt.figure(figsize=(12, 5))
        plt.subplot(131), plt.imshow(gray_raw, cmap="gray"), plt.title("Raw")
        plt.subplot(132), plt.imshow(gray, cmap="gray"), plt.title("Enhanced (CLAHE)")
        plt.subplot(133), plt.imshow(img_veins_bin, cmap="gray"), plt.title(
            "Extracted Veins"
        )
        plt.tight_layout()
        plt.show()

    return img_veins_bin


def process_single_image(image_path, output_path=None):
    """
    处理单个文件的入口函数
    """
    if not os.path.exists(image_path):
        print(f"Error: File not found at {image_path}")
        return

    print(f"Loading image: {image_path}")
    img = cv2.imread(image_path)

    if img is None:
        print("Error: Failed to read image. Check file format.")
        return

    # 调用核心算法
    # sigma=3.0 是经验值，如果你的图片分辨率很高(比如>640x480)，可能需要调大 sigma
    # result = max_curvature(img, sigma=35.0, show=True)
    # sigma14 hand 25best或者35
    # sigma20 finger
    # result = repeated_line_tracking(img, mask=None, iterations=200000, r=5, W=45)
    result = gabor_vein_extract(img, ksizes=[6], directions=8)
    if output_path:
        cv2.imwrite(output_path, result)
        print(f"Result saved to: {output_path}")


if __name__ == "__main__":
    # --- 用户配置区域 ---
    # 将下面的路径改为你电脑上实际的图片路径
    input_file = "./data/ROI/handall/1/1_0.bmp"
    output_file = "./result_vein.png"

    # 创建一个假图片用于测试（如果你没有图片的话）
    if not os.path.exists(input_file):
        print("Warning: Input file not found. Creating a dummy test image...")
        dummy = np.random.randint(50, 200, (300, 300), dtype=np.uint8)
        # 画几条线模拟静脉
        cv2.line(dummy, (50, 50), (250, 250), 30, 5)
        cv2.line(dummy, (250, 50), (50, 250), 30, 5)
        cv2.imwrite(input_file, dummy)

    # 运行处理
    process_single_image(input_file, output_file)
