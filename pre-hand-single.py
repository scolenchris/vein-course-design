import cv2
import matplotlib.pyplot as plt
import numpy as np
import traceback
import os

# 1. 引入 file_ops 模块
from utils import preprocessing, geometry, roi_extraction, enhancement, file_ops


def main_single():
    # ==========================
    # 1. 单张调试参数设置
    # ==========================
    # 修改这里为你想要测试的图片路径
    image_path = "./data/raw/hand/PV1/11/11_6.bmp"

    # ==========================
    # 新增：路径映射逻辑
    # ==========================
    # input_root: 对应 image_path 的根目录部分，用于计算相对路径
    input_root = "./data/raw/"
    # output_root: 你希望保存 ROI 的根目录
    output_root = "./data/ROI/"

    # 自动生成对应的保存路径 (例如: ./data/ROI/hand/PV1/11/11_5.bmp)
    # 并自动创建中间缺失的文件夹
    output_save_path = file_ops.get_output_path(image_path, input_root, output_root)

    print(f"Processing Single Image: {image_path}")
    print(f"Target Save Path: {output_save_path}")

    # params = {
    #     # --- 新增旋转选项 ---
    #     # 'cw'  : 顺时针 90 度
    #     # 'ccw' : 逆时针 90 度
    #     # None  : 不旋转
    #     "rotate_mode": None,
    #     "L_threshold": 250,
    #     "peaks_margin": 40,
    #     "extend_len": 50,
    #     "sharpen_margin": 15,
    #     "binary_threshold_coeff": 0.7,  # 二值化阈值系数
    #     "enhancement": {
    #         "enable_retinex": False,
    #         "enable_he": False,
    #         "enable_clahe": True,
    #         "enable_norm": True,
    #     },
    # }

    params = {
        # --- 新增旋转选项 ---
        # 'cw'  : 顺时针 90 度
        # 'ccw' : 逆时针 90 度
        # None  : 不旋转
        "rotate_mode": None,
        "L_threshold": 280,
        "peaks_margin": 40,
        "extend_len": 20,
        "sharpen_margin": 15,
        "binary_threshold_coeff": 0.7,  # 二值化阈值系数
        "enhancement": {
            "enable_retinex": False,
            "enable_he": False,
            "enable_clahe": True,
            "enable_norm": True,
        },
    }

    # 初始化变量，防止报错时变量未定义
    step2_result = None
    pref_point = None
    found_points = []
    rdf_data = []
    extra_points = []
    roi_ref_points = {}
    roi_raw = None
    roi_enhanced = None
    hand_type = "Init"

    try:
        # --- Step 1: 预处理 ---
        img_padded, step1_mask_padded = preprocessing.preprocess_hand(
            image_path,
            extend_len=params["extend_len"],
            sharpen_margin=params["sharpen_margin"],
            binary_threshold_coeff=params["binary_threshold_coeff"],
            rotate_mode=params["rotate_mode"],
        )

        # --- Step 2: 手腕剔除 ---
        step2_result, debug_info = preprocessing.wrist_culling(
            step1_mask_padded, L=params["L_threshold"]
        )
        pref_point = debug_info.get("pref_point")
        if pref_point is None:
            raise ValueError("Pref point not found")

        # --- Step 3: 特征点提取 ---
        found_points, rdf_data, ordered_contour = geometry.extract_peaks_and_valleys(
            step2_result, pref_point, margin=params["peaks_margin"]
        )

        # --- Step 4: 几何分析 ---
        hand_type, extra_points, roi_ref_points = geometry.analyze_hand_geometry(
            found_points, rdf_data, ordered_contour
        )
        print(f"Hand Type Detected: {hand_type}")

        if "Anomaly" in hand_type or not roi_ref_points:
            print(f"Warning: Geometry Analysis indicates anomaly: {hand_type}")
        else:
            # --- Step 5: ROI 提取 ---
            P1 = roi_ref_points["P1"]
            P2 = roi_ref_points["P2"]
            roi_raw, _ = roi_extraction.extract_roi_rotated(
                img_padded, P1, P2, hand_type, output_size=(500, 500)
            )

            # --- Step 6: 图像增强 ---
            roi_enhanced = enhancement.apply_enhancements(
                roi_raw, params["enhancement"]
            )

            # 保存结果 (使用上面生成的动态路径)
            cv2.imwrite(output_save_path, roi_enhanced)
            print(f"Saved enhanced ROI to: {output_save_path}")

    except Exception as e:
        print(f"Error processing image: {e}")
        traceback.print_exc()

    # ==========================
    # 可视化 (调试核心部分)
    # ==========================
    if step2_result is not None:
        plt.figure(figsize=(14, 10))

        # --- Subplot 1: 掩膜与特征点 (物理位置) ---
        plt.subplot(2, 2, 1)
        plt.title(f"Features on Mask ({hand_type})")

        vis_img = cv2.cvtColor(step2_result, cv2.COLOR_GRAY2BGR)

        if pref_point:
            cv2.circle(vis_img, pref_point, 8, (255, 0, 0), -1)
            cv2.putText(
                vis_img,
                "Pref",
                (pref_point[0] + 10, pref_point[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1,
            )

        for p in found_points:
            pt = p["point"]
            color = (0, 0, 255) if p["type"] == "peak" else (0, 255, 0)
            cv2.circle(vis_img, pt, 6, color, -1)

        for p in extra_points:
            pt = p["point"]
            cv2.circle(vis_img, pt, 6, (0, 255, 255), -1)

        if roi_ref_points:
            for key, pt in roi_ref_points.items():
                cv2.circle(vis_img, pt, 8, (255, 0, 255), -1)
                cv2.putText(
                    vis_img, key, pt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2
                )

        plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
        plt.axis("off")

        # --- Subplot 2: RDF 曲线分析 ---
        plt.subplot(2, 2, 2)
        title_color = "red" if "Anomaly" in hand_type else "black"
        plt.title("RDF Graph Analysis", color=title_color)

        if len(rdf_data) > 0:
            plt.plot(rdf_data, color="blue", alpha=0.6, label="RDF")
            # 简化了绘图逻辑，避免重复 Label
            added_labels = set()
            for p in found_points:
                lbl = "Peak" if p["type"] == "peak" else "Valley"
                style = "ro" if p["type"] == "peak" else "go"
                if lbl not in added_labels:
                    plt.plot(p["index"], p["dist"], style, label=lbl)
                    added_labels.add(lbl)
                else:
                    plt.plot(p["index"], p["dist"], style)

            if extra_points:
                plt.plot(
                    [p["index"] for p in extra_points],
                    [p["dist"] for p in extra_points],
                    "yo",
                    markeredgecolor="k",
                    label="Extra Points",
                )

            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.5)
        else:
            plt.text(0.5, 0.5, "No RDF Data", ha="center")

        # --- Subplot 3: 原始 ROI ---
        plt.subplot(2, 2, 3)
        plt.title("Raw ROI")
        if roi_raw is not None:
            plt.imshow(roi_raw, cmap="gray")
        else:
            plt.text(0.5, 0.5, "ROI Extraction Failed", ha="center")
        plt.axis("off")

        # --- Subplot 4: 增强后 ROI ---
        plt.subplot(2, 2, 4)
        plt.title("Enhanced ROI (Final)")
        if roi_enhanced is not None:
            plt.imshow(roi_enhanced, cmap="gray")
        else:
            plt.text(0.5, 0.5, "Enhancement Not Applied", ha="center")
        plt.axis("off")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main_single()
