import os
import glob
import datetime


def get_all_bmp_files(root_dir):
    """递归获取所有bmp文件路径，并统一使用正斜杠"""
    search_pattern = os.path.join(root_dir, "**", "*.bmp")
    files = glob.glob(search_pattern, recursive=True)

    # 强制将所有路径中的反斜杠替换为正斜杠
    files = [f.replace("\\", "/") for f in files]
    return files


def get_output_path(input_path, input_root, output_root):
    """
    根据输入路径生成输出路径，保持原有目录结构
    同时确保输出路径使用正斜杠
    """
    rel_path = os.path.relpath(input_path, input_root)
    out_path = os.path.join(output_root, rel_path)

    # 确保输出文件的目录存在
    out_dir = os.path.dirname(out_path)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # 统一路径分隔符为 '/'
    return out_path.replace("\\", "/")


def write_error_log(log_file_path, img_path, error_msg):
    """
    将失败信息写入日志文件
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 确保写入日志的路径也是统一格式
    clean_img_path = img_path.replace("\\", "/")

    with open(log_file_path, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] 处理失败文件: {clean_img_path}\n")
        f.write(f"    原因: {error_msg}\n")
        f.write("-" * 50 + "\n")
