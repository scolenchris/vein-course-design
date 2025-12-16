import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 假设 enhancement.py 在 utils 文件夹中
from utils import enhancement

# --- 辅助函数 ---


def load_grayscale_image(file_path):
    """
    加载图像并转换为灰度图
    """
    if not os.path.exists(file_path):
        print(f"错误：找不到文件 {file_path}")
        # 创建一个简单的测试图像以继续演示
        print("将创建一个 100x100 的测试灰度图像")
        return np.random.randint(0, 256, (100, 100), dtype=np.uint8)

    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"错误：无法读取图像文件 {file_path}")
        return None
    return img


def visualize_comparison(images_dict):
    """
    使用 Matplotlib 可视化对比图像，排版更紧凑
    """
    # 计算子图数量
    num_images = len(images_dict)

    # 确定最佳布局 (最多 4 列)
    num_cols = min(num_images, 4)
    num_rows = (num_images + num_cols - 1) // num_cols

    # 调整 figure size 的基础系数 (例如 3.0) 以实现更紧凑的显示
    base_size = 3.0
    fig, axes = plt.subplots(
        num_rows, num_cols, figsize=(base_size * num_cols, base_size * num_rows)
    )

    # 将 axes 转换为扁平数组，方便迭代
    if num_rows == 1 and num_cols == 1:
        axes = np.array([axes])
    elif num_rows == 1 or num_cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    for i, (title, img) in enumerate(images_dict.items()):
        ax = axes[i]
        ax.imshow(img, cmap="gray")

        # 调整标题字体大小，使其更适合紧凑布局
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    # 隐藏多余的子图
    for j in range(num_images, num_rows * num_cols):
        fig.delaxes(axes[j])

    # 使用 tight_layout 并设置较小的 padding，使图像紧密排列
    plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=1.0)
    plt.show()


# --- 主对比逻辑 ---


def compare_enhancement_methods(image_path="test_image.jpg"):
    """
    加载图像并对比不同的增强方法
    """
    # 1. 加载图像
    original_img = load_grayscale_image(image_path)
    if original_img is None:
        return

    # 2. 定义对比的增强配置
    enhancement_configs = {
        "Original Image": {},
        # 单独方法
        "Normalization Only": {"enable_norm": True},
        "Global HE Only": {"enable_he": True, "enable_norm": True},
        "CLAHE Only": {"enable_clahe": True, "enable_norm": True},
        "MSR Only": {"enable_retinex": True},
        # 组合方法 (Retinex 优先)
        "MSR + Global HE": {
            "enable_retinex": True,
            "enable_he": True,
            "enable_norm": True,
        },
        "MSR + CLAHE": {
            "enable_retinex": True,
            "enable_clahe": True,
            "enable_norm": True,
        },
        # 其它组合
        "Global HE + Norm": {"enable_he": True, "enable_norm": True},
        "CLAHE + Norm": {"enable_clahe": True, "enable_norm": True},
    }

    results = {}

    # 3. 处理图像并收集结果
    print("开始处理图像...")
    for title, config in enhancement_configs.items():
        if title == "Original Image":
            results[title] = original_img
            continue

        enhanced_img = enhancement.apply_enhancements(original_img, config)
        results[title] = enhanced_img
        print(f"处理完成：{title}")

    # 4. 可视化结果
    print("\n显示对比结果 (Matplotlib 窗口)...")
    visualize_comparison(results)


if __name__ == "__main__":

    # 您可以修改此路径
    # image_to_process = "./data/ROI/hand/PV2/1/1_6.bmp"
    image_to_process = "./data/ROI/finger/1/1.bmp"

    compare_enhancement_methods(image_to_process)
