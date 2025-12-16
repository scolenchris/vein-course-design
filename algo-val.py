# main.py
import argparse
import os
from model import metrics


def main():
    # 可以在这里指定默认路径，或者通过命令行参数传入
    # 假设你的数据放在当前目录下的 data 文件夹
    default_path = "./output/hand-rlt"

    parser = argparse.ArgumentParser(description="二值纹理图像匹配评估系统")
    parser.add_argument(
        "--path", type=str, default=default_path, help="包含类别子文件夹的数据集路径"
    )

    args = parser.parse_args()

    data_path = args.path

    if not os.path.exists(data_path):
        print(f"错误: 找不到路径 {data_path}")
        print("请确保数据文件夹存在，并且包含按类别分类的二值图像。")
        return

    print(f"开始处理数据集: {data_path}")

    # 1. 计算分数
    # 这一步会自动调用 model/matcher.py 里的 IoU 算法
    gen_scores, imp_scores = metrics.algo_calculate_scores(data_path)

    # 2. 绘图
    # 绘制直方图和ROC曲线
    metrics.algo_plot_evaluation_results(gen_scores, imp_scores)


if __name__ == "__main__":
    main()
