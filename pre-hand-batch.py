import cv2
import sys
import os

# 增加旋转pv2处理

# 导入 utils 包下的各个模块 (新增 enhancement)
from utils import file_ops, preprocessing, geometry, roi_extraction, enhancement


def process_single_image(img_path, output_path, params):
    """
    处理单张图片逻辑
    """
    # 1. 预处理
    img_padded, step1_mask_padded = preprocessing.preprocess_hand(
        img_path,
        extend_len=params["extend_len"],
        sharpen_margin=params["sharpen_margin"],
        binary_threshold_coeff=params["binary_threshold_coeff"],
        rotate_mode=params.get("rotate_mode", None),
    )

    # 2. 手腕剔除
    step2_result, debug_info = preprocessing.wrist_culling(
        step1_mask_padded, L=params["L_threshold"]
    )

    pref_point = debug_info.get("pref_point")
    if pref_point is None:
        raise ValueError("Pref point not found (Wrist culling failed)")

    # 3. 特征提取
    found_points, rdf_data, ordered_contour = geometry.extract_peaks_and_valleys(
        step2_result, pref_point, margin=params["peaks_margin"]
    )

    # 4. 几何分析
    hand_type, _, roi_ref_points = geometry.analyze_hand_geometry(
        found_points, rdf_data, ordered_contour
    )

    if "Anomaly" in hand_type or not roi_ref_points:
        raise ValueError(f"Geometry analysis failed: {hand_type}")

    # 5. ROI 提取
    P1 = roi_ref_points["P1"]
    P2 = roi_ref_points["P2"]
    roi_image, _ = roi_extraction.extract_roi_rotated(
        img_padded, P1, P2, hand_type, output_size=(500, 500)
    )

    # ==========================
    # 6. 图像增强 (新增步骤)
    # ==========================
    final_image = enhancement.apply_enhancements(roi_image, params["enhancement"])

    # 7. 保存结果
    cv2.imwrite(output_path, final_image)
    return True


def main():
    # ==========================
    # 调参入口
    # ==========================

    # 默认参数
    # params = {
    #    # --- 新增旋转选项 ---
    #    # 'cw'  : 顺时针 90 度
    #    # 'ccw' : 逆时针 90 度
    #    # None  : 不旋转
    #     "rotate_mode": None,
    #     "L_threshold": 120,
    #     "peaks_margin": 30,
    #     "extend_len": 50,
    #     "sharpen_margin": 15,
    #     "binary_threshold_coeff": 0.45,
    #     # 新增增强配置字典
    #     "enhancement": {
    #         "enable_retinex": False,  # MSR Retinex
    #         "enable_he": False,  # 全局直方图均衡化
    #         "enable_clahe": True,  # CLAHE (推荐开启)
    #         "enable_norm": True,  # 归一化
    #     },
    # }

    # PV2 处理参数1
    # params = {
    #     # --- 新增旋转选项 ---
    #     # 'cw'  : 顺时针 90 度
    #     # 'ccw' : 逆时针 90 度
    #     # None  : 不旋转
    #     "rotate_mode": "ccw",
    #     "L_threshold": 190,
    #     "peaks_margin": 40,
    #     "extend_len": 50,
    #     "sharpen_margin": 15,
    #     "binary_threshold_coeff": 0.8,
    #     # 新增增强配置字典
    #     "enhancement": {
    #         "enable_retinex": False,  # MSR Retinex
    #         "enable_he": False,  # 全局直方图均衡化
    #         "enable_clahe": True,  # CLAHE (推荐开启)
    #         "enable_norm": True,  # 归一化
    #     },
    # }

    # PV2 处理参数2
    # params = {
    #     # --- 新增旋转选项 ---
    #     # 'cw'  : 顺时针 90 度
    #     # 'ccw' : 逆时针 90 度
    #     # None  : 不旋转
    #     "rotate_mode": "ccw",
    #     "L_threshold": 220,
    #     "peaks_margin": 40,
    #     "extend_len": 50,
    #     "sharpen_margin": 15,
    #     "binary_threshold_coeff": 0.9,
    #     # 新增增强配置字典
    #     "enhancement": {
    #         "enable_retinex": False,  # MSR Retinex
    #         "enable_he": False,  # 全局直方图均衡化
    #         "enable_clahe": True,  # CLAHE (推荐开启)
    #         "enable_norm": True,  # 归一化
    #     },
    # }
    # PV2 处理参数3
    # params = {
    #     # --- 新增旋转选项 ---
    #     # 'cw'  : 顺时针 90 度
    #     # 'ccw' : 逆时针 90 度
    #     # None  : 不旋转
    #     "rotate_mode": "ccw",
    #     "L_threshold": 180,
    #     "peaks_margin": 40,
    #     "extend_len": 50,
    #     "sharpen_margin": 15,
    #     "binary_threshold_coeff": 0.9,
    #     # 新增增强配置字典
    #     "enhancement": {
    #         "enable_retinex": False,  # MSR Retinex
    #         "enable_he": False,  # 全局直方图均衡化
    #         "enable_clahe": True,  # CLAHE (推荐开启)
    #         "enable_norm": True,  # 归一化
    #     },
    # }

    # pv1处理参数1
    # params = {
    #     # --- 新增旋转选项 ---
    #     # 'cw'  : 顺时针 90 度
    #     # 'ccw' : 逆时针 90 度
    #     # None  : 不旋转
    #     "rotate_mode": None,
    #     "L_threshold": 120,
    #     "peaks_margin": 30,
    #     "extend_len": 50,
    #     "sharpen_margin": 15,
    #     "binary_threshold_coeff": 0.45,
    #     # 新增增强配置字典
    #     "enhancement": {
    #         "enable_retinex": False,  # MSR Retinex
    #         "enable_he": False,  # 全局直方图均衡化
    #         "enable_clahe": True,  # CLAHE (推荐开启)
    #         "enable_norm": True,  # 归一化
    #     },
    # }
    # PV1 处理参数2
    params = {
        # --- 新增旋转选项 ---
        # 'cw'  : 顺时针 90 度
        # 'ccw' : 逆时针 90 度
        # None  : 不旋转
        "rotate_mode": None,
        "L_threshold": 200,
        "peaks_margin": 40,
        "extend_len": 50,
        "sharpen_margin": 15,
        "binary_threshold_coeff": 0.7,
        # 新增增强配置字典
        "enhancement": {
            "enable_retinex": False,  # MSR Retinex
            "enable_he": False,  # 全局直方图均衡化
            "enable_clahe": True,  # CLAHE (推荐开启)
            "enable_norm": True,  # 归一化
        },
    }

    input_root = "./data/raw/hand/PV1/"
    output_root = "./data/ROI/hand/PV1/"
    log_file = "./processing_errors.log"

    if os.path.exists(log_file):
        os.remove(log_file)

    print(f"Scanning files in {input_root}...")
    file_list = file_ops.get_all_bmp_files(input_root)
    print(f"Found {len(file_list)} files.")

    success_count = 0
    fail_count = 0

    for img_path in file_list:
        output_path = file_ops.get_output_path(img_path, input_root, output_root)
        print(f"Processing: {img_path}")

        try:
            process_single_image(img_path, output_path, params)
            success_count += 1
        except Exception as e:
            error_msg = str(e)
            print(f"  [FAILED] Reason: {error_msg}")
            file_ops.write_error_log(log_file, img_path, error_msg)
            fail_count += 1
        print("-" * 30)

    print(f"\nProcessing Complete.")
    print(f"Success: {success_count}")
    print(f"Failed: {fail_count}")


if __name__ == "__main__":
    main()
