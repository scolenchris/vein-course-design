import os
import shutil
import random
import math


def split_dataset(src_roots, dest_root, split_ratio=(0.7, 0.2, 0.1)):
    """
    Args:
        src_roots: 源数据根目录列表 (e.g., ['data/ROI/hand/PV1', 'data/ROI/hand/PV2'])
        dest_root: 输出根目录 (e.g., './output/hand')
        split_ratio: (train, test, val) 比例，和应为 1
    """
    train_ratio, test_ratio, val_ratio = split_ratio

    # 1. 准备数据字典：用于合并 PV1 和 PV2 中同一个人的数据
    # 格式: { 'person_id': ['full_path_to_img1', 'full_path_to_img2', ...] }
    data_map = {}

    print("正在扫描源目录...")
    for root_path in src_roots:
        if not os.path.exists(root_path):
            print(f"警告: 路径不存在 {root_path}")
            continue

        # 遍历每个人的文件夹 (例如: root_path/1, root_path/2)
        for person_id in os.listdir(root_path):
            person_dir = os.path.join(root_path, person_id)

            if not os.path.isdir(person_dir):
                continue

            if person_id not in data_map:
                data_map[person_id] = []

            # 收集该人的所有图片
            for img_name in os.listdir(person_dir):
                if img_name.lower().endswith((".bmp", ".jpg", ".png", ".jpeg")):
                    data_map[person_id].append(os.path.join(person_dir, img_name))

    print(f"共发现 {len(data_map)} 个分类 (不同的人)。开始划分...")

    # 2. 创建输出目录结构
    sets = ["train", "test", "val"]
    for s in sets:
        os.makedirs(os.path.join(dest_root, s), exist_ok=True)

    # 3. 遍历每个人，进行打乱和分发
    total_images_count = 0

    for person_id, images in data_map.items():
        # 打乱顺序
        random.shuffle(images)
        total_count = len(images)
        total_images_count += total_count

        # 计算数量
        # 注意：如果图片太少（如5张），严格按比例可能会导致 val 为 0
        # 这里做一个简单的逻辑：优先保证 Test 和 Val 至少有 1 张（如果总数允许）

        n_test = int(total_count * test_ratio)
        n_val = int(total_count * val_ratio)

        # 针对小样本数据的特殊处理 (5张图的情况)
        if total_count >= 3:
            # 强制保证验证集和测试集至少有一张，防止训练过程中报错
            if n_test == 0:
                n_test = 1
            if n_val == 0:
                n_val = 1
            # 如果加上强制分配后超出了总数，优先削减 train
            if (n_test + n_val) >= total_count:
                n_train = 0  # 极端情况
            else:
                n_train = total_count - n_test - n_val
        else:
            # 数据极少的情况，全部放入 train
            n_train = total_count
            n_test = 0
            n_val = 0

        # 切片列表
        train_imgs = images[:n_train]
        test_imgs = images[n_train : n_train + n_test]
        val_imgs = images[n_train + n_test :]

        # 执行复制操作的辅助函数
        def copy_files(file_list, set_name):
            save_dir = os.path.join(dest_root, set_name, person_id)
            os.makedirs(save_dir, exist_ok=True)
            for file_path in file_list:
                file_name = os.path.basename(file_path)
                # 为了防止 PV1 和 PV2 有重名文件，可以在文件名前加前缀，或直接覆盖
                # 这里假设 PV1/1/a.bmp 和 PV2/1/a.bmp 是不同的，如果不确定，建议重命名
                # 下面代码保留原文件名，如有重名会覆盖
                shutil.copy2(file_path, os.path.join(save_dir, file_name))

        copy_files(train_imgs, "train")
        copy_files(test_imgs, "test")
        copy_files(val_imgs, "val")

    print("--- 处理完成 ---")
    print(f"输出目录: {dest_root}")
    print(f"处理类别数: {len(data_map)}")
    print(f"处理图片总数: {total_images_count}")
    print("文件夹结构已生成：")
    print(f"  - {os.path.join(dest_root, 'train')}")
    print(f"  - {os.path.join(dest_root, 'test')}")
    print(f"  - {os.path.join(dest_root, 'val')}")


if __name__ == "__main__":
    # 配置路径
    source_dirs = ["./data/ROI/hand/PV1", "./data/ROI/hand/PV2"]
    output_dir = "./output/handdata"

    ratios = (0.6, 0.2, 0.2)

    split_dataset(source_dirs, output_dir, ratios)
