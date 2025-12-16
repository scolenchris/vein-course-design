import cv2
import numpy as np


def extract_roi_rotated(gray_img, P1, P2, hand_type, output_size=(100, 100)):
    """
    提取旋转后的 ROI
    注意：output_size 参数现在仅用于生成异常情况下的兜底黑图，正常情况下不再对 ROI 进行强制缩放。
    """
    p1_f = np.array(P1, dtype=np.float32)
    p2_f = np.array(P2, dtype=np.float32)

    v = p2_f - p1_f
    d = np.linalg.norm(v)

    # 异常情况：无法计算方向，返回指定大小的全黑图
    if d == 0:
        return np.zeros(output_size, dtype=np.uint8), None

    u = v / d
    normal = np.array([-u[1], u[0]])

    if "Left" in hand_type:
        normal = -normal

    angle_rad = np.arctan2(v[1], v[0])
    angle_deg = np.degrees(angle_rad)
    rotation_angle = angle_deg - 90

    center = tuple(p1_f)
    M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)

    h, w = gray_img.shape
    rotated_img = cv2.warpAffine(
        gray_img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0
    )

    c1 = p1_f
    c2 = p2_f
    c3 = p2_f + normal * d
    c4 = p1_f + normal * d
    original_corners = np.array([c1, c2, c3, c4])
    ones = np.ones((original_corners.shape[0], 1))
    points_homo = np.hstack([original_corners, ones])
    rotated_corners = M.dot(points_homo.T).T

    x_coords = rotated_corners[:, 0]
    y_coords = rotated_corners[:, 1]

    min_x = int(np.floor(np.min(x_coords)))
    max_x = int(np.ceil(np.max(x_coords)))
    min_y = int(np.floor(np.min(y_coords)))
    max_y = int(np.ceil(np.max(y_coords)))

    min_x = max(0, min_x)
    min_y = max(0, min_y)
    max_x = min(w, max_x)
    max_y = min(h, max_y)

    roi_crop = rotated_img[min_y:max_y, min_x:max_x]
    # crop_h, crop_w = roi_crop.shape[:2]
    # crop_size = (crop_w, crop_h)
    # print(crop_size)
    # ==========================
    # 修改：删去了缩放操作
    # ==========================
    # 原代码：
    try:
        roi_final = cv2.resize(roi_crop, output_size, interpolation=cv2.INTER_AREA)
        return roi_final, M
    except Exception as e:
        return np.zeros(output_size, dtype=np.uint8), M

    # 新代码：
    # 检查裁切是否有效（防止空切片）
    if roi_crop.size == 0:
        return np.zeros(output_size, dtype=np.uint8), M

    return roi_crop, M
