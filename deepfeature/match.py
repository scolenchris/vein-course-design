import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# 引入之前的模块
from model import VeinNet, VeinNetV2
from utils import get_device

# ================= 配置区域 =================
# 模型权重路径 (请修改为你训练出来的最佳模型文件名)
MODEL_PATH = "./" + "output/best_VeinNetV2_hand_(128, 128)_20251210_171733.pth"

# 数据相关配置
INPUT_SIZE = (128, 128)
DATATYPE = "hand"  # 或者 'hand'
DATA_DIR = f"./output/{DATATYPE}data/train"  # 使用测试集进行验证
BATCH_SIZE = 4  # 推理时batch size可以随意

# 选定用于测试的那张图片的索引 (例如第0张图)
QUERY_INDEX = 0
# ===========================================


def load_model(model_path, device):
    """加载训练好的模型"""
    # 注意：这里需要根据你实际训练的模型类来实例化 (VeinNet 或 VeinNetV2)
    model = VeinNetV2(num_classes=15).to(device)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model weights from {model_path}")
    else:
        print(
            f"Error: Model file '{model_path}' not found. Please train the model first."
        )
        sys.exit(1)

    model.eval()  # 开启评估模式
    return model


def get_test_loader():
    """加载测试集数据"""
    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory '{DATA_DIR}' not found.")
        sys.exit(1)

    test_transform = transforms.Compose(
        [transforms.Resize(INPUT_SIZE), transforms.ToTensor()]
    )

    # ImageFolder 会自动加载图片并附带标签
    test_dataset = torchvision.datasets.ImageFolder(
        root=DATA_DIR, transform=test_transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False
    )

    return test_loader, test_dataset


def extract_all_features(model, loader, device):
    """提取数据集中所有图片的特征和标签"""
    all_features = []
    all_labels = []

    print("Extracting features from test set...")
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)

            # 调用我们在 model.py 中定义的 extract_feature 方法
            # 输出形状: [batch_size, 64]
            features = model.extract_feature(inputs)

            # 归一化特征向量 (对于余弦相似度很重要)
            # 这一步将向量长度变为1，之后计算点积就等于余弦相似度
            features = torch.nn.functional.normalize(features, p=2, dim=1)

            all_features.append(features.cpu())
            all_labels.append(labels)

    # 拼接成一个大的 Tensor
    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    print(f"Extracted features shape: {all_features.shape}")
    return all_features, all_labels


def main():
    device = get_device()
    print(f"Using device: {device}")

    # 1. 准备数据和模型
    test_loader, test_dataset = get_test_loader()
    model = load_model(MODEL_PATH, device)

    # 2. 提取所有特征库
    features_bank, labels_bank = extract_all_features(model, test_loader, device)

    # 3. 选定一个 Query (样本)
    if QUERY_INDEX >= len(features_bank):
        print(f"Error: Query Index {QUERY_INDEX} is out of range.")
        return

    query_feat = features_bank[QUERY_INDEX].unsqueeze(0)  # [1, 64]
    query_label = labels_bank[QUERY_INDEX].item()

    print(f"Selected Query Image Index: {QUERY_INDEX}, True Label: {query_label}")

    # 4. 计算相似度 (Cosine Similarity)
    # 因为特征已经归一化，所以直接矩阵乘法就是余弦相似度
    # [1, 64] * [N, 64]^T -> [1, N]
    similarities = torch.mm(query_feat, features_bank.t()).squeeze(0).numpy()

    # 5. 分离同类和异类
    same_class_scores = []
    diff_class_scores = []

    for idx, score in enumerate(similarities):
        # 跳过自己和自己的比对 (相似度肯定为1，没有意义)
        if idx == QUERY_INDEX:
            continue

        if labels_bank[idx].item() == query_label:
            same_class_scores.append(score)
        else:
            diff_class_scores.append(score)

    same_class_scores = np.array(same_class_scores)
    diff_class_scores = np.array(diff_class_scores)

    print(f"Count - Same Class Pairs: {len(same_class_scores)}")
    print(f"Count - Diff Class Pairs: {len(diff_class_scores)}")
    if len(same_class_scores) > 0:
        print(f"Avg Same Class Similarity: {same_class_scores.mean():.4f}")
    print(f"Avg Diff Class Similarity: {diff_class_scores.mean():.4f}")

    # 6. 绘图：相似度曲线 (Sorted Similarity Curve)
    plt.figure(figsize=(12, 5))

    # 子图1: 排序后的相似度曲线 (能够清晰看到两者分离的程度)
    plt.subplot(1, 2, 1)
    if len(same_class_scores) > 0:
        plt.plot(
            np.sort(same_class_scores)[::-1],
            label="Same Class (Genuine)",
            color="blue",
            linewidth=2,
        )
    plt.plot(
        np.sort(diff_class_scores)[::-1],
        label="Diff Class (Imposter)",
        color="red",
        linewidth=2,
    )
    plt.title("Sorted Similarity Curves")
    plt.xlabel("Sample Index (Sorted)")
    plt.ylabel("Cosine Similarity Score")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    # 子图2: 相似度分布直方图 (更直观地看到两个分布的重叠情况)
    plt.subplot(1, 2, 2)
    if len(same_class_scores) > 0:
        plt.hist(
            same_class_scores,
            bins=10,
            alpha=0.7,
            label="Same Class",
            color="blue",
            density=True,
        )
    plt.hist(
        diff_class_scores,
        bins=50,
        alpha=0.5,
        label="Diff Class",
        color="red",
        density=True,
    )
    plt.title("Similarity Score Distribution")
    plt.xlabel("Cosine Similarity Score")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()

    # 保存结果图片
    save_file = f"similarity_analysis_idx{QUERY_INDEX}.png"
    plt.savefig(save_file)
    print(f"Result plot saved to {save_file}")
    plt.show()


if __name__ == "__main__":
    main()
