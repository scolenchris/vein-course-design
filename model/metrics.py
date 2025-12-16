# model/metrics.py
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import random
from itertools import combinations
from sklearn.metrics import roc_curve, auc
import model.matcher as matcher


def load_all_features(feature_dir):
    """加载所有 .npy 特征文件"""
    features_db = {}
    if not os.path.exists(feature_dir):
        print(f"路径不存在: {feature_dir}")
        return features_db

    for class_folder in os.listdir(feature_dir):
        class_path = os.path.join(feature_dir, class_folder)
        if os.path.isdir(class_path):
            features_db[class_folder] = []
            npy_files = glob.glob(os.path.join(class_path, "*.npy"))
            for npy_file in npy_files:
                try:
                    feat = np.load(npy_file)
                    features_db[class_folder].append(feat)
                except Exception as e:
                    print(f"加载出错 {npy_file}: {e}")
    return features_db


def calculate_scores(feature_dir, extractor):
    """计算 Genuine 和 Imposter 分数"""
    print("正在加载特征库...")
    db = load_all_features(feature_dir)
    classes = list(db.keys())

    genuine_scores = []
    imposter_scores = []

    print(f"共发现 {len(classes)} 个类别。正在进行匹配计算...")

    # 1. 类内匹配 (Genuine)
    for cls in classes:
        feats = db[cls]
        if len(feats) < 2:
            continue
        for f1, f2 in combinations(feats, 2):
            score = extractor.match_histograms(f1, f2)
            genuine_scores.append(score)

    # 2. 类间匹配 (Imposter) - 简化版：每类取第一个样本互比
    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            if len(db[classes[i]]) > 0 and len(db[classes[j]]) > 0:
                f1 = db[classes[i]][0]
                f2 = db[classes[j]][0]
                score = extractor.match_histograms(f1, f2)
                imposter_scores.append(score)

    return np.array(genuine_scores), np.array(imposter_scores)


def plot_evaluation_results(genuine_scores, imposter_scores, max_possible_score):
    """绘制分数分布和 ROC 曲线"""
    plt.figure(figsize=(12, 5))

    # Plot 1: Histogram
    plt.subplot(1, 2, 1)
    plt.hist(
        genuine_scores, bins=50, alpha=0.7, label="Genuine", color="green", density=True
    )
    plt.hist(
        imposter_scores, bins=50, alpha=0.7, label="Imposter", color="red", density=True
    )
    plt.title("Similarity Score Distribution")
    plt.xlabel(f"Score (Max {max_possible_score})")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: ROC
    y_true = np.concatenate(
        [np.ones(len(genuine_scores)), np.zeros(len(imposter_scores))]
    )
    y_scores = np.concatenate([genuine_scores, imposter_scores])

    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    roc_auc = auc(fpr, tpr)

    fnr = 1 - tpr
    eer_index = np.nanargmin(np.abs(fnr - fpr))
    eer = fpr[eer_index]

    plt.subplot(1, 2, 2)
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="navy")
    plt.scatter(fpr[eer_index], tpr[eer_index], color="red", label=f"EER = {eer:.2%}")
    plt.xlabel("FAR (False Positive Rate)")
    plt.ylabel("TAR (True Positive Rate)")
    plt.title(f"ROC Curve (EER={eer:.2%})")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"评估完成 -> EER: {eer:.2%}, AUC: {roc_auc:.4f}")


# def algo_plot_evaluation_results(genuine_scores, imposter_scores):
#     """绘制分数分布和 ROC 曲线"""
#     if len(genuine_scores) == 0 or len(imposter_scores) == 0:
#         print("分数数据不足，无法绘图。")
#         return

#     plt.figure(figsize=(12, 5))

#     # ==========================================
#     # Plot 1: Histogram (Similarity Score Distribution)
#     # ==========================================
#     plt.subplot(1, 2, 1)

#     # 为了让 0-0.2 的范围内显示得足够细致，我们显式定义 bins
#     # 如果只用 bins=50，matplotlib 会在 0-1 全局分50份，导致 0-0.2 只有10个柱子，很难看。
#     # 这里我们强制在 0 到 0.2 之间切分 50 个柱子。
#     bins = np.linspace(0, 0.2, 50)

#     plt.hist(
#         genuine_scores,
#         bins=bins,
#         alpha=0.7,
#         label="Genuine",
#         color="green",
#         density=True,
#     )
#     plt.hist(
#         imposter_scores,
#         bins=bins,
#         alpha=0.7,
#         label="Imposter",
#         color="red",
#         density=True,
#     )

#     plt.title("Similarity Score Distribution")
#     plt.xlabel("Score (IoU)")
#     # 强制限制 x 轴显示范围
#     plt.xlim(0, 0.2)
#     plt.legend()
#     plt.grid(True, alpha=0.3)

#     # ==========================================
#     # Plot 2: ROC Curve
#     # ==========================================
#     y_true = np.concatenate(
#         [np.ones(len(genuine_scores)), np.zeros(len(imposter_scores))]
#     )
#     y_scores = np.concatenate([genuine_scores, imposter_scores])

#     fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
#     roc_auc = auc(fpr, tpr)

#     fnr = 1 - tpr
#     eer_index = np.nanargmin(np.abs(fnr - fpr))
#     eer = fpr[eer_index]

#     plt.subplot(1, 2, 2)
#     plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.4f}")
#     plt.plot([0, 1], [0, 1], linestyle="--", color="navy")
#     plt.scatter(fpr[eer_index], tpr[eer_index], color="red", label=f"EER = {eer:.2%}")

#     plt.xlabel("FAR (False Positive Rate)")
#     plt.ylabel("TAR (True Positive Rate)")
#     plt.title(f"ROC Curve (EER={eer:.2%})")
#     plt.legend(loc="lower right")
#     plt.grid(True, alpha=0.3)

#     plt.tight_layout()
#     plt.show()

#     print(f"评估完成 -> EER: {eer:.2%}, AUC: {roc_auc:.4f}")


def algo_plot_evaluation_results(genuine_scores, imposter_scores):
    """绘制分数分布和 ROC 曲线 (修正版：自适应坐标轴范围)"""
    if len(genuine_scores) == 0 or len(imposter_scores) == 0:
        print("分数数据不足，无法绘图。")
        return

    plt.figure(figsize=(12, 5))

    # ==========================================
    # Plot 1: Histogram (Similarity Score Distribution)
    # ==========================================
    plt.subplot(1, 2, 1)

    # 1. 获取数据的最大值，决定画图范围
    all_scores = np.concatenate([genuine_scores, imposter_scores])
    max_score = np.max(all_scores)

    # 设定上限：如果最大分超过0.2，就用最大分+0.05，否则默认用0.25（为了美观）
    # IoU的理论最大值是 1.0
    upper_limit = max(0.25, max_score * 1.1)
    if upper_limit > 1.0:
        upper_limit = 1.0

    # 2. 动态生成 bins
    # 在 0 到 upper_limit 之间切分 50 份
    bins = np.linspace(0, upper_limit, 50)

    plt.hist(
        genuine_scores,
        bins=bins,
        alpha=0.7,
        label="Genuine",
        color="green",
        density=True,
    )
    plt.hist(
        imposter_scores,
        bins=bins,
        alpha=0.7,
        label="Imposter",
        color="red",
        density=True,
    )

    plt.title("Similarity Score Distribution")
    plt.xlabel("Score (IoU)")

    # 3. 动态设置 X 轴范围
    plt.xlim(0, upper_limit)

    plt.legend()
    plt.grid(True, alpha=0.3)

    # ==========================================
    # Plot 2: ROC Curve
    # ==========================================
    y_true = np.concatenate(
        [np.ones(len(genuine_scores)), np.zeros(len(imposter_scores))]
    )
    y_scores = np.concatenate([genuine_scores, imposter_scores])

    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    roc_auc = auc(fpr, tpr)

    fnr = 1 - tpr
    # 避免除零或空值的保护
    if len(fpr) > 0:
        eer_index = np.nanargmin(np.abs(fnr - fpr))
        eer = fpr[eer_index]
    else:
        eer = 0

    plt.subplot(1, 2, 2)
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="navy")
    plt.scatter(fpr[eer_index], tpr[eer_index], color="red", label=f"EER = {eer:.2%}")

    plt.xlabel("FAR (False Positive Rate)")
    plt.ylabel("TAR (True Positive Rate)")
    plt.title(f"ROC Curve (EER={eer:.2%})")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"评估完成 -> EER: {eer:.2%}, AUC: {roc_auc:.4f}")
    # 打印一下最大分数，验证我的猜想
    print(
        f"调试信息 -> Genuine最高分: {np.max(genuine_scores):.4f}, Imposter最高分: {np.max(imposter_scores):.4f}"
    )


def load_all_images(data_dir):
    """加载所有图片路径"""
    images_db = {}
    if not os.path.exists(data_dir):
        print(f"路径不存在: {data_dir}")
        return images_db

    # 遍历每个类别的文件夹
    for class_folder in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_folder)
        if os.path.isdir(class_path):
            # 支持常见图片格式
            img_files = []
            for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp"]:
                img_files.extend(glob.glob(os.path.join(class_path, ext)))

            if img_files:
                images_db[class_folder] = img_files

    return images_db


def algo_calculate_scores(data_dir):
    """计算 Genuine (同类) 和 Imposter (异类) 分数"""
    print("正在扫描数据集...")
    db = load_all_images(data_dir)
    classes = list(db.keys())

    if len(classes) == 0:
        print("未找到任何类别数据，请检查路径。")
        return np.array([]), np.array([])

    genuine_scores = []
    imposter_scores = []

    print(f"共发现 {len(classes)} 个类别。正在进行特征提取与匹配计算...")

    # 为了避免重复读取IO，我们可以选择先缓存图片数据，
    # 或者在循环中实时读取（节省内存）。这里采用实时读取。

    # 1. 类内匹配 (Genuine)
    print("正在计算类内匹配 (Genuine)...")
    for cls in classes:
        img_paths = db[cls]
        if len(img_paths) < 2:
            continue

        # 限制每类最大匹配对数，防止数据量过大跑太久
        pairs = list(combinations(img_paths, 2))
        if len(pairs) > 100:
            pairs = random.sample(pairs, 100)

        for p1, p2 in pairs:
            try:
                img1 = matcher.read_binary_image(p1)
                img2 = matcher.read_binary_image(p2)
                score = matcher.calculate_iou_score(img1, img2)
                genuine_scores.append(score)
            except Exception as e:
                print(f"处理出错: {e}")

    # 2. 类间匹配 (Imposter)
    print("正在计算类间匹配 (Imposter)...")
    # 为了画出好看的ROC，我们需要足够的负样本。
    # 策略：随机抽取不同类别的图片对进行比对。
    num_imposter_pairs = max(len(genuine_scores), 1000)  # 保持正负样本数量级相当

    cnt = 0
    while cnt < num_imposter_pairs:
        # 随机选两个不同的类
        c1, c2 = random.sample(classes, 2)
        # 从这两个类随机选一张图
        if not db[c1] or not db[c2]:
            continue

        p1 = random.choice(db[c1])
        p2 = random.choice(db[c2])

        try:
            img1 = matcher.read_binary_image(p1)
            img2 = matcher.read_binary_image(p2)
            score = matcher.calculate_iou_score(img1, img2)
            imposter_scores.append(score)
            cnt += 1
        except Exception as e:
            pass

    return np.array(genuine_scores), np.array(imposter_scores)
