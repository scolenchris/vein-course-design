import os
import shutil
import random


def split_dataset(source_root, target_root, split_ratio=(0.7, 0.2, 0.1)):
    """
    按照比例划分数据集
    :param source_root: 源数据集根目录 (e.g., 'data/ROI/finger')
    :param target_root: 输出数据集根目录 (e.g., './output/fingerdata')
    :param split_ratio: 划分比例 tuple (train, test, val), 和必须为1
    """

    # 1. 检查比例和是否为1
    if abs(sum(split_ratio) - 1.0) > 1e-5:
        print("错误: 划分比例之和必须为 1")
        return

    train_ratio, test_ratio, val_ratio = split_ratio

    # 定义输出的三个子目录名
    dir_names = ["train", "test", "val"]

    # 2. 获取所有类别（即所有人的文件夹名为类别名）
    if not os.path.exists(source_root):
        print(f"错误: 源目录 {source_root} 不存在")
        return

    classes = [
        d
        for d in os.listdir(source_root)
        if os.path.isdir(os.path.join(source_root, d))
    ]

    print(f"检测到 {len(classes)} 个类别（人员），开始处理...")

    # 设置随机种子，保证如果需要复现可以固定 (可选)
    random.seed(42)

    for class_name in classes:
        # 当前类别的源路径
        class_dir = os.path.join(source_root, class_name)
        # 获取该类别下所有图片文件
        images = [
            f
            for f in os.listdir(class_dir)
            if f.lower().endswith((".bmp", ".jpg", ".png", ".jpeg"))
        ]

        # 3. 打乱图片顺序
        random.shuffle(images)

        # 4. 计算划分的数量
        total_num = len(images)
        train_num = int(total_num * train_ratio)
        test_num = int(total_num * test_ratio)
        # 验证集取剩余的，以防止因取整导致总数不对
        val_num = total_num - train_num - test_num

        # 切片列表
        train_imgs = images[:train_num]
        test_imgs = images[train_num : train_num + test_num]
        val_imgs = images[train_num + test_num :]

        # 5. 复制文件到目标目录
        # 定义 datasets 和对应图片的映射
        splits = [("train", train_imgs), ("test", test_imgs), ("val", val_imgs)]

        for split_name, split_imgs in splits:
            # 目标路径: output/fingerdata/train/class_name/
            # 注意：这里保留了 class_name 文件夹，这是分类任务的标准格式
            target_dir = os.path.join(target_root, split_name, class_name)

            # 如果目录不存在则创建
            os.makedirs(target_dir, exist_ok=True)

            for img in split_imgs:
                src_file = os.path.join(class_dir, img)
                dst_file = os.path.join(target_dir, img)
                shutil.copy2(src_file, dst_file)  # copy2 保留文件元数据

        # 打印少量进度信息
        # print(f"处理类别 {class_name}: 总数{total_num} -> 训练{len(train_imgs)} / 测试{len(test_imgs)} / 验证{len(val_imgs)}")

    print("-" * 30)
    print(f"处理完成！数据集已输出到: {target_root}")
    print(
        f"划分比例: 训练集 {train_ratio*100}% | 测试集 {test_ratio*100}% | 验证集 {val_ratio*100}%"
    )


if __name__ == "__main__":
    # 配置路径
    # 注意：Windows路径建议使用 r'' 或者双反斜杠 \\
    src_path = "./data/ROI/finger"
    dst_path = "./output/fingerdata"

    split_dataset(src_path, dst_path, split_ratio=(0.6, 0.2, 0.2))
