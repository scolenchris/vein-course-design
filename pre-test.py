import cv2
import numpy as np
import math


# ---------- 1. 工具函数 ----------
def otsu(gray: np.ndarray):
    """返回 Otsu 阈值"""
    ret, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return ret


def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def find_max_idx(arr, margin):
    """在环形邻域内找极大值索引"""
    n = len(arr)
    for i in range(n):
        is_max = True
        for k in range(-margin, margin + 1):
            if arr[(i + k) % n] > arr[i]:
                is_max = False
                break
        if is_max:
            return i
    return 0


def find_min_idx(arr, margin):
    """在环形邻域内找极小值索引"""
    n = len(arr)
    for i in range(n):
        is_min = True
        for k in range(-margin, margin + 1):
            if arr[(i + k) % n] < arr[i]:
                is_min = False
                break
        if is_min:
            return i
    return 0


# ---------- 2. 最大轮廓提取 ----------
def get_max_region(bin_img: np.ndarray) -> np.ndarray:
    """
    返回面积最大的轮廓点集 (N,1,2) 格式
    若检测不到轮廓返回 None
    """
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    max_cnt = max(contours, key=cv2.contourArea)
    return max_cnt


# ---------- 3. 找 9 个指峰/指谷 ----------
def get_square_two_points(cnt: np.ndarray, imgW: int) -> list:
    """
    输入最大轮廓，返回 [square_pt0, square_pt1] 两个关键点
    square_pt0 靠近腕部，square_pt1 靠近指根
    """
    # 1) 把轮廓 reshape 成 (N,2)
    pts = cnt.reshape(-1, 2).astype(np.int32)
    n = len(pts)

    # 2) 找最右侧（靠近图像右边缘）的起始点
    start_idx = 0
    max_x = 0
    for i, (x, y) in enumerate(pts):
        if x >= imgW - 5 and y > pts[start_idx][1]:
            start_idx = i
            max_x = x

    # 3) 计算到“中心参考点”的距离序列
    center = np.array([imgW, pts[start_idx][1]], dtype=float)
    dis = np.array([distance(p, center) for p in pts])

    # 4) 在距离序列中找 9 个极值（峰/谷交替）
    finger_pts = []  # 9 个点
    idx_seq = []  # 对应索引
    margin = 30  # 邻域半径，可调
    search_order = list(range(start_idx, start_idx + n))
    num = 0
    for idx in search_order:
        idx = idx % n
        if num % 2 == 0:  # 找极大
            if idx == find_max_idx(dis, margin):
                finger_pts.append(pts[idx])
                idx_seq.append(idx)
                num += 1
        else:  # 找极小
            if idx == find_min_idx(dis, margin):
                finger_pts.append(pts[idx])
                idx_seq.append(idx)
                num += 1
        if num == 9:
            break
    if num < 9:
        raise RuntimeError("未能检测到 9 个指峰/谷，请调整 margin 或检查输入")

    # 5) 根据伪代码逻辑计算两个方形关键点
    def middle(p1, p2):
        return ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)

    # 简单区分手型左右
    if distance(finger_pts[1], finger_pts[3]) > distance(finger_pts[5], finger_pts[7]):
        # 右手
        flen = idx_seq[3] - idx_seq[2]
        p_else = pts[(idx_seq[2] - flen) % n]
        pt1 = middle(finger_pts[3], p_else)

        flen = idx_seq[8] - idx_seq[7]
        p_else = pts[(idx_seq[8] + flen) % n]
        pt0 = middle(finger_pts[7], p_else)
    else:
        # 左手
        flen = idx_seq[1] - idx_seq[0]
        p_else = pts[(idx_seq[0] - flen) % n]
        pt1 = middle(finger_pts[1], p_else)

        flen = idx_seq[6] - idx_seq[5]
        p_else = pts[(idx_seq[6] + flen) % n]
        pt0 = middle(finger_pts[5], p_else)

    return [np.array(pt0, dtype=np.int32), np.array(pt1, dtype=np.int32)]


# ---------- 4. 旋转 + ROI 截取 ----------
def get_palm_roi(gray: np.ndarray, sq_pts: list) -> np.ndarray:
    """
    根据两个关键点，对灰度图旋转校正并截取正方形 ROI
    """
    p0, p1 = sq_pts
    # 1) 计算旋转角
    dx = float(p1[0] - p0[0])
    dy = float(p1[1] - p0[1])
    angle = math.atan2(dx, -dy) * 180 / math.pi  # 负号让指尖朝上

    # 2) 以 p0 为中心旋转
    h, w = gray.shape
    M = cv2.getRotationMatrix2D(tuple(p0), angle, 1.0)
    rotated = cv2.warpAffine(gray, M, (w, h), borderValue=0)

    # 3) 截取正方形 ROI
    side = int(distance(p0, p1))
    x0, y0 = p0
    # 边界保护
    y1 = max(y0 - side, 0)
    y2 = min(y0, h)
    x1 = max(x0, 0)
    x2 = min(x0 + side, w)
    if x2 <= x1 or y2 <= y1:
        raise RuntimeError("ROI 区域为空，请检查关键点")
    roi = rotated[y1:y2, x1:x2]
    return roi


# def refined_mask(gray: np.ndarray) -> np.ndarray:
#     """
#     输入：单通道手掌图
#     输出：边缘光滑、内部无空洞的 0/255 mask
#     """
#     # 1) 保边滤波：去纹理同时保留边缘
#     gray = cv2.bilateralFilter(gray, 9, 75, 75)

#     # 2) Otsu 粗分割
#     th = otsu(gray)
#     _, mask = cv2.threshold(gray, th, 255, cv2.THRESH_BINARY)

#     # 3) 空洞填充（Flood-Fill 法）
#     h, w = mask.shape[:2]
#     flood = mask.copy()
#     cv2.floodFill(flood, None, (0, 0), 255)  # 背景填白
#     hole = cv2.bitwise_not(flood)  # 黑色区域即孔洞
#     mask = cv2.bitwise_or(mask, hole)  # 把孔洞补回

#     # 4) 边缘平滑：Guided Filter（速度≈双边滤波，边缘更锋利）
#     mask = cv2.ximgproc.guidedFilter(gray, mask, 15, 1e-4)  # 需要 opencv-contrib
#     _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

#     # 5) 轻量级形态学：先闭运算补缝隙，再开运算去毛刺
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
#     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
#     return mask


# ---------- 5. 主入口 ----------
def get_roi_img(palm_bgr: np.ndarray) -> np.ndarray:
    """
    输入原始 BGR 手掌图，返回 ROI（BGR）
    中间弹出前背景分割结果
    """
    gray = cv2.cvtColor(palm_bgr, cv2.COLOR_BGR2GRAY)

    # 1) Otsu 二值化
    th = otsu(gray)
    _, bin_img = cv2.threshold(gray, th, 255, cv2.THRESH_BINARY)

    # 2) 形态学去噪
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel, iterations=2)

    # >>> 可视化前背景分割 <<<
    cv2.imshow("前背景分割（按任意键继续）", bin_img)
    cv2.waitKey(0)
    cv2.destroyWindow("前背景分割（按任意键继续）")

    # 后续流程不变
    cnt = get_max_region(bin_img)
    if cnt is None:
        raise RuntimeError("未能提取到手部轮廓")
    sq_pts = get_square_two_points(cnt, bin_img.shape[1])
    roi_gray = get_palm_roi(gray, sq_pts)
    roi_bgr = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR)
    return roi_bgr


# ---------- 6. 简易测试 ----------
if __name__ == "__main__":

    pic = "./data/raw/hand/PV1/1/1_0.bmp"

    img = cv2.imread(pic)
    if img is None:
        raise FileNotFoundError(pic)

    roi = get_roi_img(img)
    cv2.imshow("ROI", roi)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
