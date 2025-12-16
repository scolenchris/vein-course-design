import cv2
import numpy as np


def sharpen_cutoff_fingers(binary_mask, pad_width=40, extend_len=30, margin=15):
    """
    修剪手指轮廓
    Args:
        margin: 容差 (原代码中硬编码为15)
        extend_len: 延伸长度
    """
    h, w = binary_mask.shape
    padded_mask = cv2.copyMakeBorder(
        binary_mask,
        pad_width,
        pad_width,
        pad_width,
        pad_width,
        cv2.BORDER_CONSTANT,
        value=0,
    )
    contours, _ = cv2.findContours(
        padded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return padded_mask

    raw_contour = max(contours, key=cv2.contourArea).squeeze()
    if raw_contour.ndim == 1:
        raw_contour = raw_contour.reshape(-1, 2)

    new_contour = []
    x_min, x_max = pad_width, pad_width + w - 1

    # 使用传入的 margin 参数
    search_margin = margin

    i = 0
    while i < len(raw_contour):
        pt = raw_contour[i]
        x, y = pt[0], pt[1]
        on_left = abs(x - x_min) <= search_margin
        is_border = on_left

        if not is_border:
            new_contour.append(pt)
            i += 1
            continue

        segment_indices = [i]
        j = i + 1
        while j < len(raw_contour):
            curr_idx = j
            cx, cy = raw_contour[curr_idx]
            con_left = abs(cx - x_min) <= search_margin and on_left
            if con_left:
                segment_indices.append(curr_idx)
                j += 1
            else:
                break

        if len(segment_indices) > 5:
            start_pt = raw_contour[segment_indices[0]]
            end_pt = raw_contour[segment_indices[-1]]
            mid_x = (start_pt[0] + end_pt[0]) // 2
            mid_y = (start_pt[1] + end_pt[1]) // 2
            tip_point = [mid_x, mid_y]

            if on_left:
                tip_point[0] -= extend_len  # 使用传入的 extend_len
            new_contour.append(np.array(tip_point))
        else:
            for idx in segment_indices:
                new_contour.append(raw_contour[idx])
        i = j

    final_sharpened_mask = np.zeros_like(padded_mask)
    if len(new_contour) > 0:
        new_contour_np = np.array(new_contour).reshape((-1, 1, 2)).astype(np.int32)
        cv2.drawContours(final_sharpened_mask, [new_contour_np], -1, 255, cv2.FILLED)

    return final_sharpened_mask


def preprocess_hand(
    image_path,
    extend_len,
    sharpen_margin,
    binary_threshold_coeff=0.45,
    rotate_mode=None,
):
    """
    预处理入口
    Args:
        image_path: 图片路径
        extend_len: 切手指时的延伸长度
        sharpen_margin: 切手指轮廓容差
        binary_threshold_coeff: 二值化阈值系数
        rotate_mode: 旋转模式 ('cw': 顺时针90度, 'ccw': 逆时针90度, None: 不旋转)
    """
    # 1. 读取图片
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"无法找到图像: {image_path}")

    # 2. 【新增】内存中旋转逻辑 (不改变原文件)
    if rotate_mode == "cw":
        # 顺时针 90 度
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif rotate_mode == "ccw":
        # 逆时针 90 度
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # 如果是 None 或者其他值，则不旋转

    # 3. 继续原有处理流程
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    ret3, th3_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    new_threshold = ret3 * binary_threshold_coeff
    _, binary_final = cv2.threshold(blur, new_threshold, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    binary_final = cv2.morphologyEx(binary_final, cv2.MORPH_OPEN, kernel, iterations=1)

    temp_mask = np.zeros_like(binary_final)
    contours, _ = cv2.findContours(
        binary_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if contours:
        max_cnt = max(contours, key=cv2.contourArea)
        cv2.drawContours(temp_mask, [max_cnt], -1, 255, cv2.FILLED)
    else:
        temp_mask = binary_final.copy()

    pad_w = 200
    final_mask_padded = sharpen_cutoff_fingers(
        temp_mask, pad_width=pad_w, extend_len=extend_len, margin=sharpen_margin
    )

    img_padded = cv2.copyMakeBorder(
        img, pad_w, pad_w, pad_w, pad_w, cv2.BORDER_CONSTANT, value=0
    )

    return img_padded, final_mask_padded


def wrist_culling(binary_image, L):
    """手腕剔除"""
    if len(binary_image.shape) == 3:
        gray = cv2.cvtColor(binary_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = binary_image.copy()

    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    img_height, img_width = binary.shape

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return binary, {}

    max_contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(max_contour)
    if M["m00"] != 0:
        center_x = int(M["m10"] / M["m00"])
    else:
        x, y, w, h = cv2.boundingRect(max_contour)
        center_x = x + w // 2

    cut_line_x = min(center_x + L, img_width - 1)
    result = binary.copy()
    result[:, cut_line_x:] = 0

    intersection_column = binary[:, cut_line_x]
    white_indices = np.where(intersection_column == 255)[0]

    pref_point = None
    if len(white_indices) > 0:
        intersect_up_y = white_indices[0]
        intersect_down_y = white_indices[-1]
        pref_y = (intersect_up_y + intersect_down_y) // 2
        pref_point = (cut_line_x, pref_y)

    debug_info = {"pref_point": pref_point}
    return result, debug_info
