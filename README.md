# 机器视觉课设 - 静脉识别实验 (Vein Recognition Experiment)

本项目是一个完整的静脉识别系统实验平台，涵盖了从原始图像预处理（ROI提取）、数据集划分、传统特征提取算法（最大曲率、RLT、Gabor）、LBP纹理特征分析到深度学习（VeinNet）识别的全流程。

## 📁 目录结构说明

* `data/`: 存放原始数据和处理后的数据（需自行建立或配置路径）
* `modules/`: 传统图像处理算法模块 (Gabor, RLT, MaxCurvature)
* `utils/`: 通用工具包 (几何计算, ROI提取, 图像增强, 文件操作)
* `model/`: 传统匹配模型与评估指标 (LBP, IoU Matcher, Metrics)
* `deepfeature/`: 深度学习相关代码 (模型定义, 训练, 匹配)
* `record/`: 存放运行过程中的中间结果图
* `*.py`: 各个阶段的执行脚本

---

## 🚀 快速开始

### 1. 环境准备

请确保安装了以下 Python 库：

```bash
pip install numpy opencv-python matplotlib scikit-learn scikit-image torch torchvision

```

### 2. 图像预处理 (ROI 提取)

将原始采集的图像转换为感兴趣区域 (ROI) 图像。

#### **手指静脉预处理**

* **脚本**: `pre-finger-batch.py`
* **功能**: 批量处理手指静脉图像，包括边缘检测、旋转矫正、ROI裁剪和CLAHE增强。
* **配置**: 修改文件底部的 `INPUT_DIR` 和 `OUTPUT_DIR`。
```bash
python pre-finger-batch.py

```



#### **手背静脉预处理**

* **脚本**: `pre-hand-batch.py`
* **功能**: 针对手背静脉（PV1/PV2数据集），包含手腕剔除、指缝关键点定位、基于几何特征的旋转和裁剪。
* **配置**: 修改 `main` 函数中的 `input_root` 和 `params`（支持配置旋转模式、增强算法等）。
```bash
python pre-hand-batch.py

```


* **单张调试**: 使用 `pre-hand-single.py` 可视化单张图片的处理流程（关键点、RDF曲线、掩膜）。

#### **图像增强算法对比**

* **脚本**: `cmp_enhance.py`
* **功能**: 对比不同增强算法（HE, CLAHE, Retinex 等）的效果。
```bash
python cmp_enhance.py

```



---

### 3. 数据集管理

#### **数据完整性检查**

* **脚本**: `datacheck.py`
* **功能**: 检查 ROI 提取后是否存在缺失文件，结果生成至 `missing_files_list.txt`。

#### **数据集融合**

* **脚本**: `hand-all-gen.py`
* **功能**: 将 `PV1` 和 `PV2` 两个子文件夹的数据合并到 `handall` 目录中。

#### **数据集划分 (Train/Val/Test)**

* **脚本**:
* `datasplit-finger.py`: 划分手指静脉数据集。
* `datasplit-hand.py`: 划分手背静脉数据集。


* **说明**: 默认比例为 7:2:1 或 6:2:2，脚本会自动按人员ID隔离划分，防止数据泄露。

---

### 4. 传统算法特征提取

使用传统计算机视觉算法提取静脉纹理（二值化图像）。

#### **批量提取特征**

* **脚本**: `algo-datagen-main.py`
* **功能**: 遍历 ROI 数据集，生成静脉纹理图。
* **算法选择**: 在代码中 `process_batch` 函数内取消注释对应行：
1. `max_curvature_extract`: 最大曲率法（稳定性好）
2. `repeated_line_tracking`: 重复线追踪法 (RLT)
3. `gabor_vein_extract`: Gabor 滤波法


* **运行**:
```bash
python algo-datagen-main.py

```



#### **算法性能评估**

* **脚本**: `algo-val.py`
* **功能**: 基于二值图像的 IoU (Intersection over Union) 进行匹配，绘制 ROC 曲线并计算 EER。
* **使用**:
```bash
python algo-val.py --path ./output/hand-gabor

```



---

### 5. LBP 纹理特征分析

基于局部二值模式 (LBP) 的特征提取与直方图匹配。

* **脚本**: `main-lbp.py`
* **配置**: 修改文件头部的 `TASK_ID` 来执行不同任务。
* `TASK_ID = 1`: **单图可视化**（查看 LBP 编码图和直方图）
* `TASK_ID = 2`: **批量特征提取**（将图片转换为 `.npy` 特征文件）
* `TASK_ID = 3`: **双图匹配测试**（计算两张图片的相似度）
* `TASK_ID = 4`: **全集评估**（计算 ROC 和 EER）



---

### 6. 深度学习 (Deep Learning)

基于 PyTorch 实现的 CNN 模型 (VeinNet/VeinNetV2) 进行特征学习。

#### **模型训练**

* **脚本**: `deepfeature/train.py`
* **功能**: 加载划分好的数据集（Train/Val），训练 VeinNet 模型。
* **输出**: 训练日志 (TensorBoard) 和最佳模型权重 (`.pth`) 保存在 `output/`。
* **配置**: 修改 `INPUT_SIZE`, `BATCH_SIZE`, `DATATYPE` 等参数。
```bash
python deepfeature/train.py

```



#### **深度特征匹配与分析**

* **脚本**: `deepfeature/match.py`
* **功能**: 加载训练好的模型，提取测试集特征，计算余弦相似度，并绘制同类/异类分布图。
* **配置**: 修改 `MODEL_PATH` 指向训练好的 `.pth` 文件。
```bash
python deepfeature/match.py

```



---

## 🛠️ 常见问题

1. **路径错误**: 大部分脚本的输入输出路径定义在文件底部的 `if __name__ == "__main__":` 块中，请根据你的实际目录结构进行修改。
2. **OpenCV 报错**: 如果遇到 `cv2.error`，通常是图片路径不对导致读取为 `None`，请检查 `datacheck.py` 的结果。
3. **手背ROI失败**: `pre-hand` 系列脚本依赖几何关键点检测，如果图片背景太杂或光照太差可能导致检测失败，失败列表会记录在日志中。
