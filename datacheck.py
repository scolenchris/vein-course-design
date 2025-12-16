import os
import sys

# 导入你现有的工具包
from utils import file_ops


def check_completeness():
    # ==========================
    # 配置路径
    # ==========================
    # 原始数据根目录 (Source)
    raw_root = "./data/raw/hand/"

    # ROI 结果根目录 (Target)
    roi_root = "./data/ROI/hand/"

    print("========================================")
    print("开始检测数据集完整性...")
    print(f"原始目录 (Raw): {raw_root}")
    print(f"结果目录 (ROI): {roi_root}")
    print("========================================")

    # 1. 获取所有文件的绝对路径
    # 注意：get_all_bmp_files 返回的是完整路径
    print("正在扫描文件...")
    raw_files_full = file_ops.get_all_bmp_files(raw_root)
    roi_files_full = file_ops.get_all_bmp_files(roi_root)

    # 2. 转换为相对路径集合 (Set) 以便比对
    # 例如：将 "./data/raw/hand/PV1/1/1_1.bmp" 转换为 "PV1/1/1_1.bmp"
    raw_rel_set = set()
    for f in raw_files_full:
        # relpath 计算相对路径，replace 确保 Windows/Linux 路径分隔符统一为 '/'
        rel = os.path.relpath(f, raw_root).replace("\\", "/")
        raw_rel_set.add(rel)

    roi_rel_set = set()
    for f in roi_files_full:
        rel = os.path.relpath(f, roi_root).replace("\\", "/")
        roi_rel_set.add(rel)

    # 3. 计算差集 (Raw 中有，但 ROI 中没有的)
    missing_files = raw_rel_set - roi_rel_set

    # 排序以便查看
    missing_list = sorted(list(missing_files))

    # 4. 输出统计结果
    print(f"\n[统计信息]")
    print(f"原始图片总数: {len(raw_rel_set)}")
    print(f"ROI 图片总数: {len(roi_rel_set)}")
    print(f"缺失图片数量: {len(missing_list)}")

    print("-" * 40)

    if len(missing_list) == 0:
        print("\n✅ 恭喜！数据集完全完整，没有缺失文件。")
    else:
        print(f"\n❌ 发现 {len(missing_list)} 个文件处理缺失 (Raw中存在但ROI中不存在):")
        print("建议检查 process_errors.log 查看失败原因。\n")

        for i, f in enumerate(missing_list, 1):
            print(f"{i}. {f}")

        # 可选：将缺失列表保存到文件，方便后续只处理这些文件
        with open("missing_files_list.txt", "w", encoding="utf-8") as f_out:
            for f in missing_list:
                f_out.write(f + "\n")
        print(f"\n[提示] 缺失文件列表已保存至: missing_files_list.txt")


if __name__ == "__main__":
    check_completeness()
