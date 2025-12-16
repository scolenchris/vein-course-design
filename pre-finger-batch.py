import cv2
import numpy as np
from scipy.signal import savgol_filter
import math
import os


class FingerVeinROI_BatchProcessor:
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height

        # --- 区域界限定义 ---
        # 1. 右边界限: 只处理左边 2/3
        self.process_ratio = 2 / 3
        # 2. 左边界限: 屏蔽左边 100 像素
        self.crop_x_start = 100
        # 计算右边的像素位置
        self.crop_x_end = int(self.width * self.process_ratio)  # 约 426

    def process_image(self, img_path, save_path, target_size=(326, 220)):
        """
        处理单张图片：读取 -> 提取ROI -> 增强 -> 缩放 -> 保存
        返回: (bool, message) - 成功状态和简短信息
        """
        # 1. 读取与初始化
        img = cv2.imread(img_path, 0)
        if img is None:
            return False, "Read Error"

        # 如果原图尺寸不符，先Resize原图
        if img.shape[1] != self.width or img.shape[0] != self.height:
            img = cv2.resize(img, (self.width, self.height))

        # 2. 获取 Mask (包含左右强制屏蔽)
        img_blur = cv2.GaussianBlur(img, (7, 7), 0)
        finger_mask = self._get_finger_mask(img_blur)

        # 3. 边缘检测
        roi_mask = finger_mask[:, self.crop_x_start : self.crop_x_end]
        if cv2.countNonZero(roi_mask) == 0:
            return False, "No Finger Detected"

        upper_edge, lower_edge = self._detect_edges_from_mask(roi_mask)

        # 4. 计算中线 & 旋转角度
        midline_points = (upper_edge + lower_edge) / 2.0
        angle, _, _ = self._calculate_angle(midline_points)

        # 5. 旋转图像 (矫正水平)
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        img_corrected = cv2.warpAffine(
            img,
            M,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        # 6. 计算裁剪边界
        y_top_limit, y_bottom_limit = self._get_rotated_y_limits(
            upper_edge, lower_edge, M
        )

        # --- 安全检查 ---
        roi_height = y_bottom_limit - y_top_limit

        # 检查: 高度是否小于 180 (过细或检测错误)
        if y_bottom_limit <= y_top_limit or roi_height < 180:
            return False, f"Invalid Height ({roi_height})"

        # 7. 执行最终裁剪 (ROI)
        final_roi = img_corrected[
            y_top_limit:y_bottom_limit, self.crop_x_start : self.crop_x_end
        ]

        # 8. CLAHE 增强
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        final_enhanced = clahe.apply(final_roi)

        # 9. 统一缩放到指定尺寸 (326x220)
        final_resized = cv2.resize(
            final_enhanced, target_size, interpolation=cv2.INTER_CUBIC
        )

        # 10. 保存
        cv2.imwrite(save_path, final_resized)
        return True, "Success"

    def _get_rotated_y_limits(self, upper, lower, M):
        pts_upper = []
        pts_lower = []
        for i in range(len(upper)):
            x = i + self.crop_x_start
            pts_upper.append([x, upper[i]])
            pts_lower.append([x, lower[i]])

        pts_upper = np.array(pts_upper, dtype=np.float32).reshape(-1, 1, 2)
        pts_lower = np.array(pts_lower, dtype=np.float32).reshape(-1, 1, 2)

        pts_upper_new = cv2.transform(pts_upper, M)
        pts_lower_new = cv2.transform(pts_lower, M)

        y_top = int(np.max(pts_upper_new[:, :, 1]))
        y_bottom = int(np.min(pts_lower_new[:, :, 1]))
        return y_top, y_bottom

    def _get_finger_mask(self, img):
        rth, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        rth, binary = cv2.threshold(img, rth * 0.95, 255, cv2.THRESH_BINARY)
        h, w = binary.shape
        binary[:, int(w * self.process_ratio) :] = 0
        binary[:, : self.crop_x_start] = 0

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 5))
        eroded = cv2.erode(binary, kernel, iterations=1)
        contours, _ = cv2.findContours(
            eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return binary

        max_cnt = max(contours, key=cv2.contourArea)
        temp_mask = np.zeros_like(binary)
        cv2.drawContours(temp_mask, [max_cnt], -1, 255, thickness=cv2.FILLED)

        clean_mask = cv2.dilate(
            temp_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (35, 10)), iterations=1
        )
        clean_mask = cv2.bitwise_and(clean_mask, binary)
        clean_mask = cv2.morphologyEx(
            clean_mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)),
        )
        return clean_mask

    def _detect_edges_from_mask(self, mask):
        h, w = mask.shape
        if w == 0:
            return np.zeros(1), np.zeros(1)

        upper = np.argmax(mask > 0, axis=0)
        lower_flipped = np.argmax(mask[::-1, :] > 0, axis=0)
        lower = h - 1 - lower_flipped

        valid = np.any(mask > 0, axis=0)
        if np.any(valid):
            upper[~valid] = int(np.mean(upper[valid]))
            lower[~valid] = int(np.mean(lower[valid]))

        for x in range(1, w):
            if upper[x] < upper[x - 1] - 10:
                upper[x] = upper[x - 1]
            if lower[x] > lower[x - 1] + 10:
                lower[x] = lower[x - 1]

        if w > 31:
            try:
                upper = savgol_filter(upper, 31, 2).astype(int)
                lower = savgol_filter(lower, 31, 2).astype(int)
            except:
                pass
        return upper, lower

    def _calculate_angle(self, midline_points):
        x = np.arange(len(midline_points))
        y = midline_points
        slope, intercept = np.polyfit(x, y, 1)
        return math.degrees(math.atan(slope)), slope, intercept


def batch_process_dataset(input_root, output_root, log_file="skipped_images.txt"):
    processor = FingerVeinROI_BatchProcessor()

    total_files = 0
    processed_count = 0
    skipped_count = 0

    # 存储被跳过的路径
    skipped_files_list = []

    print(f"Start processing from: {input_root}")
    print(f"Output destination: {output_root}")

    for root, dirs, files in os.walk(input_root):
        for file in files:
            if file.lower().endswith((".bmp", ".jpg", ".png", ".jpeg")):
                total_files += 1

                # 1. 组合路径
                src_path_raw = os.path.join(root, file)

                # 2. 修复路径分隔符 (将反斜杠替换为正斜杠)
                src_path = src_path_raw.replace("\\", "/")

                # 3. 构建相对路径与目标路径
                # 注意：relpath 可能也会引入反斜杠，需要再次修复
                relative_path = os.path.relpath(src_path, input_root).replace("\\", "/")
                dst_path = os.path.join(output_root, relative_path).replace("\\", "/")

                # 4. 确保目录存在
                dst_dir = os.path.dirname(dst_path)
                if not os.path.exists(dst_dir):
                    os.makedirs(dst_dir, exist_ok=True)

                # 5. 处理图像
                # print(f"Processing: {src_path} ...", end="\r")

                success, msg = processor.process_image(
                    src_path, dst_path, target_size=(326, 220)
                )

                if success:
                    processed_count += 1
                else:
                    skipped_count += 1
                    print(f"\n[SKIP] {msg}: {src_path}")
                    skipped_files_list.append(src_path)

    # 保存跳过的路径到 TXT
    if skipped_files_list:
        with open(log_file, "w", encoding="utf-8") as f:
            for path in skipped_files_list:
                f.write(path + "\n")
        print(f"\nSaved {len(skipped_files_list)} skipped paths to '{log_file}'")

    print(f"\n--- Batch Process Finished ---")
    print(f"Total Found: {total_files}")
    print(f"Processed:   {processed_count}")
    print(f"Skipped:     {skipped_count}")


if __name__ == "__main__":
    # 配置路径
    INPUT_DIR = "./data/raw/finger"
    OUTPUT_DIR = "./data/ROI/finger"

    # 运行
    batch_process_dataset(INPUT_DIR, OUTPUT_DIR)
