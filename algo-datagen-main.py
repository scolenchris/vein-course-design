import os
import cv2
import time

# 导入我们在 modules 文件夹里写的函数
from modules.vein_algo import max_curvature_extract
from modules.gabor import gabor_vein_extract
from modules.rlt import repeated_line_tracking


def process_batch(input_root_dir, output_root_dir):
    """
    递归遍历输入文件夹，处理所有BMP图片，并按原结构保存到输出文件夹
    """
    # 记录统计信息
    total_files = 0
    success_count = 0
    start_time = time.time()

    print(f"开始处理: {input_root_dir} -> {output_root_dir}")

    # os.walk 会递归遍历所有子目录
    for root, dirs, files in os.walk(input_root_dir):
        for filename in files:
            # 检查文件扩展名 (忽略大小写)
            if filename.lower().endswith(".bmp"):
                total_files += 1

                # 1. 构建完整的输入路径
                input_path = os.path.join(root, filename)

                # 2. 构建保持结构的输出路径
                # 计算当前文件相对于输入根目录的相对路径 (例如: subdir/image.bmp)
                relative_path = os.path.relpath(root, input_root_dir)
                # 组合输出目录 (例如: output_root/subdir)
                target_dir = os.path.join(output_root_dir, relative_path)

                # 确保输出目录存在，不存在则创建
                os.makedirs(target_dir, exist_ok=True)

                # 组合最终输出文件路径
                output_path = os.path.join(target_dir, filename)

                # 3. 处理图片
                try:
                    # 读取
                    img = cv2.imread(input_path)
                    if img is None:
                        print(f"[跳过] 无法读取文件: {input_path}")
                        continue

                    # --- 调用模块中的接口 ---
                    # 你可以在这里调整 sigma 参数
                    # result_img = max_curvature_extract(img, sigma=35.0)  # curve
                    # result_img = repeated_line_tracking(
                    #     img, mask=None, iterations=200000, r=5, W=45
                    # )  # 重复线追踪
                    result_img = gabor_vein_extract(
                        img, ksizes=[4], directions=8
                    )  # Gabor
                    # ---------------------

                    # 保存
                    cv2.imwrite(output_path, result_img)
                    success_count += 1

                    # 打印进度 (每10张或单行覆盖打印)
                    print(
                        f"[{success_count}] 已处理: {os.path.join(relative_path, filename)}"
                    )

                except Exception as e:
                    print(f"[错误] 处理文件 {input_path} 时出错: {e}")

    end_time = time.time()
    duration = end_time - start_time
    print("-" * 30)
    print(f"处理完成！")
    print(f"耗时: {duration:.2f} 秒")
    print(f"总文件数: {total_files}")
    print(f"成功保存: {success_count}")


if __name__ == "__main__":
    # --- 配置区域 ---
    # 输入文件夹路径 (请改为你的实际路径)
    INPUT_DIR = "./data/ROI/handall"

    # 输出文件夹路径 (会自动创建)
    OUTPUT_DIR = "./output/hand-gabor"

    # 检查输入路径是否存在
    if not os.path.exists(INPUT_DIR):
        print(f"错误: 输入文件夹不存在 -> {INPUT_DIR}")
    else:
        process_batch(INPUT_DIR, OUTPUT_DIR)
