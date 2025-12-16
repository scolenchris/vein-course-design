import os
import shutil
from pathlib import Path

# 设置路径
base_path = Path("./data/ROI/hand")
pv1_path = base_path / "PV1"
pv2_path = base_path / "PV2"
output_path = base_path / "handall"

# 确保输出目录存在
output_path.mkdir(parents=True, exist_ok=True)

# 获取 PV1 和 PV2 中的子文件夹
pv1_dirs = {d.name: d for d in pv1_path.iterdir() if d.is_dir()}
pv2_dirs = {d.name: d for d in pv2_path.iterdir() if d.is_dir()}

# 合并同名文件夹
all_folder_names = set(pv1_dirs.keys()).union(pv2_dirs.keys())

for folder_name in all_folder_names:
    src_dirs = []
    if folder_name in pv1_dirs:
        src_dirs.append(pv1_dirs[folder_name])
    if folder_name in pv2_dirs:
        src_dirs.append(pv2_dirs[folder_name])

    dest_dir = output_path / folder_name
    dest_dir.mkdir(exist_ok=True)

    for src_dir in src_dirs:
        for item in src_dir.iterdir():
            dest_item = dest_dir / item.name
            if item.is_file():
                shutil.copy2(item, dest_item)
            elif item.is_dir():
                if dest_item.exists():
                    shutil.rmtree(dest_item)
                shutil.copytree(item, dest_item)

print("✅ 合并完成！结果保存在：", output_path.resolve())
