import cv2
import numpy as np
import random


def repeated_line_tracking(raw_img, mask=None, iterations=200000, r=1, W=29):
    """
    重复线跟踪算法 (Repeated Line Tracking) Python 实现
    对应 C++: RepeatedLineTracking 函数

    :param raw_img: 输入图像 (numpy array, BGR或灰度)
    :param mask: ROI 掩膜 (numpy array, 可选), 必须与原图大小一致
    :param iterations: 随机种子的迭代次数 (对应 C++ main 中的 200000)
    :param r: 截面检测的步长 (对应 C++ r=1)
    :param W: 截面宽度 (对应 C++ W=29)，必须为奇数
    :return: 轨迹图像 (numpy array, uint8)
    """

    # 1. 预处理与类型转换
    if len(raw_img.shape) == 3:
        src = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
    else:
        src = raw_img.copy()

    # 转换为 float64 并归一化到 0-1 (匹配 C++: src.convertTo(src, CV_64F, 1.0/255.0))
    src = src.astype(np.float64) / 255.0
    h, w = src.shape

    # 处理 Mask
    if mask is None:
        mask_u = np.ones((h, w), dtype=np.uint8)
    else:
        mask_u = mask.copy().astype(np.uint8)

    # 结果累加器 (Tr), 使用 int32 防止溢出，最后转回 uint8
    Tr = np.zeros((h, w), dtype=np.int32)

    # 参数检查
    if W % 2 == 0:
        raise ValueError("FAIL: RepeatedLineTracking - W cannot be even")

    # 2. 计算几何参数
    # cvRound 近似
    ro = int(np.round(r * np.sqrt(2) / 2))
    hW = (W - 1) // 2
    hWo = int(np.round(hW * np.sqrt(2) / 2))

    # 3. 处理 Mask 边界 (Omit unreachable borders)
    # Top/Bottom
    limit_y = r + hW
    mask_u[0 : limit_y + 1, :] = 0
    mask_u[h - limit_y - 1 :, :] = 0

    # Left/Right
    limit_x = r + hW
    mask_u[:, 0 : limit_x + 1] = 0
    mask_u[:, w - limit_x - 1 :] = 0

    # 4. 随机种子生成
    # C++ 中是循环 iterations 次，每次随机取点，如果 mask 有效则通过。
    # 为了 Python 效率，我们先获取所有有效区域的坐标，然后随机采样。
    valid_ys, valid_xs = np.where(mask_u > 0)
    if len(valid_xs) == 0:
        return np.zeros((h, w), dtype=np.uint8)

    num_valid = len(valid_xs)
    # 生成随机索引
    rand_indices = np.random.randint(0, num_valid, size=iterations)

    # 概率参数
    p_lr = 0.5
    p_ud = 0.25

    # 邻域映射表 (对应 C++ bla 矩阵)
    # 索引顺序: (dx=-1, dy=-1) -> 0, (-1, 0) -> 1 ...
    # C++ 代码逻辑: tmp = (dx+1)*3 + (dy+1)
    # mapping maps linear index to (dx, dy)
    # idx: 0->(-1,-1), 1->(-1,0), 2->(-1,1), 3->(0,-1), 4->(0,0), 5->(0,1), 6->(1,-1), 7->(1,0), 8->(1,1)
    # 但 C++ bla 定义似乎是手动指定的顺序?
    # check C++: bla.at<char>(0, tmp) is x, (1, tmp) is y.
    # C++ bla 初始化:
    # -1, -1 (idx 0)
    # -1, 0  (idx 1)
    # ...
    # 这正好对应 (dx, dy) 的顺序，只要 dx 是主要行序。
    # 我们直接在循环里算 dx, dy 即可，不需要查表。

    # 5. 主循环 (Iterate through all starting points)
    # 注意：这是计算密集型循环，Python 中运行较慢
    for i in range(iterations):
        idx = rand_indices[i]
        xc = valid_xs[idx]
        yc = valid_ys[idx]

        # 确定初始移动方向属性
        # Going left (-1) or right (1)?
        Dlr = 1 if random.random() < 0.5 else -1
        # Going up (-1) or down (1)?
        Dud = (
            1 if random.random() < 0.5 else -1
        )  # C++ code: if(rng.uniform(0,2)) -> 0 or 1.

        # Initialize locus-position table Tc (当前轨迹的访问记录)
        # 使用 set 记录当前路径访问过的点，比新建全尺寸 Mat 快
        Tc = set()

        Vl = 1.0

        while Vl > 0:
            # Determine moving candidate point set Nr (3x3)
            # Nr 用 3x3 boolean 数组表示
            Nr = np.zeros((3, 3), dtype=bool)

            rand_val = random.random()  # 0.0 to 1.0

            # C++ uniform(0,101)/100.0 produces 0.00 to 1.00

            if rand_val < p_lr:
                # Going left or right (row priority)
                # Dlr is -1 (left) or 1 (right).
                # grid x index: 1 + Dlr -> if left(0), if right(2)
                col_idx = 1 + Dlr
                Nr[:, col_idx] = True  # Set whole column

            elif rand_val < (p_lr + p_ud):
                # Going up or down (col priority)
                # Dud is -1 (up) or 1 (down)
                # grid y index: 1 + Dud -> if up(0), if down(2)
                row_idx = 1 + Dud
                Nr[row_idx, :] = True  # Set whole row
            else:
                # Any direction
                Nr[:, :] = True
                Nr[1, 1] = False  # Exclude center

            # 收集候选点 (Nc)
            Nc = []  # List of (x, y)

            # 遍历 3x3 邻域
            # dx, dy in [-1, 0, 1]
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    # Nr 索引是 (dy+1, dx+1) 注意行列关系
                    # C++: Nr.at(Point(dx+1, dy+1)) -> x是col(dx), y是row(dy)
                    if Nr[dy + 1, dx + 1]:
                        nxt_x = xc + dx
                        nxt_y = yc + dy

                        # 检查 Mask 和 是否已访问 (Tc)
                        if (nxt_x, nxt_y) not in Tc:
                            if mask_u[nxt_y, nxt_x] > 0:
                                Nc.append((nxt_x, nxt_y))

            if not Nc:
                Vl = -1
                continue

            # Detect dark line direction (Calculate Vdepths)
            best_idx = -1
            max_val = -99999.0

            for k in range(len(Nc)):
                ncp_x, ncp_y = Nc[k]
                val = 0.0

                # Horizontal or Vertical?
                if ncp_y == yc:
                    # Horizontal plane
                    yp = ncp_y
                    xp = ncp_x + r if ncp_x > xc else ncp_x - r
                    # Check vertical profile
                    val = src[yp + hW, xp] - 2 * src[yp, xp] + src[yp - hW, xp]

                elif ncp_x == xc:
                    # Vertical plane
                    xp = ncp_x
                    yp = ncp_y + r if ncp_y > yc else ncp_y - r
                    # Check horizontal profile
                    val = src[yp, xp + hW] - 2 * src[yp, xp] + src[yp, xp - hW]

                # Oblique directions
                elif (ncp_x > xc and ncp_y < yc) or (ncp_x < xc and ncp_y > yc):
                    # Diagonal up / (Top-Right or Bottom-Left)
                    if ncp_x > xc and ncp_y < yc:  # Top Right
                        xp = ncp_x + ro
                        yp = ncp_y - ro
                    else:  # Bottom Left
                        xp = ncp_x - ro
                        yp = ncp_y + ro

                    # Profile direction is \
                    val = (
                        src[yp - hWo, xp - hWo]
                        - 2 * src[yp, xp]
                        + src[yp + hWo, xp + hWo]
                    )

                else:
                    # Diagonal down \ (Top-Left or Bottom-Right)
                    if ncp_x < xc and ncp_y < yc:  # Top Left
                        xp = ncp_x - ro
                        yp = ncp_y - ro
                    else:  # Bottom Right
                        xp = ncp_x + ro
                        yp = ncp_y + ro

                    # Profile direction is /
                    val = (
                        src[yp + hWo, xp - hWo]
                        - 2 * src[yp, xp]
                        + src[yp - hWo, xp + hWo]
                    )

                # Find max
                if val > max_val:
                    max_val = val
                    best_idx = k

            # Update State
            Tc.add((xc, yc))  # Mark current as visited
            Tr[yc, xc] += 1  # Increment global accumulator

            # Move to best candidate
            xc, yc = Nc[best_idx]
            Vl = max_val

    # 6. 生成结果
    # C++代码简单地将 Tr copyTo dst。Tr 是累加值。
    # 通常 RLT 结果需要归一化显示，或者直接截断。
    # C++ 中 Tr 是 Mat(CV_8U) 但在循环中 ++，这依赖于 uchar 的溢出或者饱和。
    # 这里我们将 int32 的累加结果截断到 255 并转为 uint8。

    # 简单的截断策略：
    img_dst = np.clip(Tr, 0, 255).astype(np.uint8)
    img_dst = post_process_binary(img_dst)  # 进行二值化处理，实验single可以删掉
    return img_dst


def post_process_binary(rlt_result):
    """
    对 RLT 结果进行二值化处理
    """
    # 1. 归一化 (Normalization)
    # RLT 的结果可能是暗淡的（如果迭代次数少），归一化可以让最亮的部分变成 255
    norm_img = cv2.normalize(rlt_result, None, 0, 255, cv2.NORM_MINMAX)

    # 2. 二值化 (Otsu's Thresholding)
    # 0, 255 是范围，THRESH_OTSU 会忽略第一个参数(0)，自动寻找最佳阈值
    ret, binary_img = cv2.threshold(
        norm_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    ret, binary_img = cv2.threshold(norm_img, ret * 0.3, 255, cv2.THRESH_BINARY)
    # 打印自动计算出的阈值，方便调试
    # print(f"Otsu's calculated threshold: {ret}")

    # 3. 形态学去噪 (Morphological Operations) - 可选
    # 如果结果中有许多细小的白色噪点，可以用“开运算”去除
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # 开运算：先腐蚀后膨胀，去除小白点
    binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)

    # 或者：如果静脉有断裂，可以用“闭运算”连接
    binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)

    return binary_img
