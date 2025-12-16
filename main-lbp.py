# main.py
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 导入自定义模块
from model.LBP import LBPFeatureExtractor
from model.metrics import calculate_scores, plot_evaluation_results

# ================= 配置区域 (Configuration) =================

# 选择任务 ID:
# 1: 单张图片可视化 (调试用)
# 2: 批量特征提取 (生成 .npy)
# 3: 两张图片特征匹配 (测试用)
# 4: 整体数据集评估 (ROC/EER)
TASK_ID = 4

# 路径设置
# INPUT_DIR = "./data/ROI/handall"  # 原始图片文件夹
# OUTPUT_DIR = "./output/hand-lbp"  # 特征保存文件夹
# SAMPLE_IMG = "./data/ROI/handall/1/1_0.bmp"  # 任务1用
# SAMPLE_NPY_1 = "./output/hand-lbp/1/1_0.npy"  # 任务3用
# SAMPLE_NPY_2 = "./output/hand-lbp/2/2_0.npy"  # 任务3用

# finger路径
INPUT_DIR = "./data/ROI/finger"  # 原始图片文件夹
OUTPUT_DIR = "./output/finger-lbp"  # 特征保存文件夹
SAMPLE_IMG = "./data/ROI/finger/1/1.bmp"  # 任务1用
# SAMPLE_IMG = "./data/ROI/hand/PV1/1/1_0.bmp"  # 任务1用
SAMPLE_NPY_1 = "./output/finger-lbp/1/1.npy"  # 任务3用
SAMPLE_NPY_2 = "./output/finger-lbp/2/2.npy"  # 任务3用


# 模型参数
# LBP_PARAMS = {
#     "radius": 10,
#     "n_points": 32,
#     "grid_x": 8,
#     "grid_y": 8,
#     "target_size": (256, 256),
# }
# finger最佳
LBP_PARAMS = {
    "radius": 10,
    "n_points": 32,
    "grid_x": 8,
    "grid_y": 8,
    "target_size": (256, 256),
}
# hand最佳
# LBP_PARAMS = {
#     "radius": 8,
#     "n_points": 16,
#     "grid_x": 8,
#     "grid_y": 8,
#     "target_size": (256, 256),
# }
# ================= 任务实现函数 =================


def task_viz_single(extractor, img_path):
    """任务1：可视化单张图片的 LBP 特征"""
    if not os.path.exists(img_path):
        print(f"[Error] 图片不存在: {img_path}")
        return

    print(f"正在处理: {img_path}")
    feat, orig, lbp_map = extractor.extract_features(img_path, return_lbp_map=True)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(orig, cmap="gray")
    plt.title("Preprocessed")
    plt.subplot(1, 3, 2)
    plt.imshow(lbp_map, cmap="gray")
    plt.title("LBP Map")
    plt.subplot(1, 3, 3)
    plt.plot(feat, color="k", lw=0.5)
    plt.title("Feature Hist")
    plt.tight_layout()
    plt.show()


def task_batch_extract(extractor, input_root, output_root):
    """任务2：批量提取特征并保存"""
    print(f"开始批量处理: {input_root} -> {output_root}")
    input_path = Path(input_root)
    output_path = Path(output_root)
    count = 0

    for root, dirs, files in os.walk(input_path):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".bmp", ".tif")):
                src_file = Path(root) / file
                # 保持目录结构
                rel_path = src_file.relative_to(input_path)
                target_folder = output_path / rel_path.parent
                target_folder.mkdir(parents=True, exist_ok=True)

                try:
                    feat = extractor.extract_features(str(src_file))
                    save_name = target_folder / (src_file.stem + ".npy")
                    np.save(str(save_name), feat)
                    count += 1
                    if count % 100 == 0:
                        print(f"已处理 {count} 张...")
                except Exception as e:
                    print(f"跳过 {file}: {e}")
    print(f"处理完成，共 {count} 张。")


def task_match_pair(extractor, npy1, npy2):
    """任务3：匹配两个特征文件"""
    if not (os.path.exists(npy1) and os.path.exists(npy2)):
        print("特征文件不存在，请先运行任务2。")
        return

    f1 = np.load(npy1)
    f2 = np.load(npy2)
    score = extractor.match_histograms(f1, f2)
    max_score = extractor.grid_x * extractor.grid_y
    sim_percent = (score / max_score) * 100

    print(f"文件A: {os.path.basename(npy1)}")
    print(f"文件B: {os.path.basename(npy2)}")
    print(f"匹配分数: {score:.4f} / {max_score}")
    print(f"相似度: {sim_percent:.2f}%")


def task_evaluate_dataset(extractor, feature_dir):
    """任务4：评估整个数据集 (ROC/EER)"""
    gen, imp = calculate_scores(feature_dir, extractor)
    if len(gen) == 0 or len(imp) == 0:
        print("样本不足，无法评估。")
        return

    max_score = extractor.grid_x * extractor.grid_y
    plot_evaluation_results(gen, imp, max_score)


# ================= 主入口 (Entry Point) =================

if __name__ == "__main__":
    # 1. 初始化模型
    lbp_model = LBPFeatureExtractor(**LBP_PARAMS)

    # 2. 根据 TASK_ID 分发任务
    if TASK_ID == 1:
        print(">>> 运行任务 1: 单图可视化")
        task_viz_single(lbp_model, SAMPLE_IMG)

    elif TASK_ID == 2:
        print(">>> 运行任务 2: 批量特征提取")
        task_batch_extract(lbp_model, INPUT_DIR, OUTPUT_DIR)

    elif TASK_ID == 3:
        print(">>> 运行任务 3: 特征匹配测试")
        task_match_pair(lbp_model, SAMPLE_NPY_1, SAMPLE_NPY_2)

    elif TASK_ID == 4:
        print(">>> 运行任务 4: 数据集性能评估")
        task_evaluate_dataset(lbp_model, OUTPUT_DIR)

    else:
        print("未知的 TASK_ID，请检查配置。")
