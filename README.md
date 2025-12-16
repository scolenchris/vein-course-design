# vein-course-design
机器视觉课设-静脉识别实验

## 文件使用说明

### 1. 预处理模块（ROI提取）

#### **手指静脉预处理**（批量处理）
```bash
# 文件：pre-finger-batch.py
# 功能：批量处理手指静脉图像，提取ROI区域
# 配置：修改INPUT_DIR和OUTPUT_DIR路径
# 运行：直接执行
python pre-finger-batch.py
```

#### **手部静脉预处理**（批量处理）
```bash
# 文件：pre-hand-batch.py
# 功能：批量处理手部静脉图像（PV1/PV2数据集）
# 配置：调整参数params，设置input_root和output_root
# 注意：需要先创建utils模块
python pre-hand-batch.py
```

#### **手部静脉预处理**（单张调试）
```bash
# 文件：pre-hand-single.py
# 功能：单张图像调试，可视化处理过程
# 配置：修改image_path为测试图片路径
# 需要：utils模块支持
python pre-hand-single.py
```

### 2. 数据集合并与划分

#### **数据集合并**
```bash
# 文件：hand-all-gen.py
# 功能：合并PV1和PV2手部数据集
# 运行：直接执行，自动生成合并后的handall文件夹
python hand-all-gen.py
```

#### **手指数据集划分**
```bash
# 文件：datasplit-finger.py
# 功能：划分手指静脉数据集为train/test/val
# 配置：修改src_path和dst_path
# 参数：split_ratio调整划分比例
python datasplit-finger.py
```

#### **手部数据集划分**
```bash
# 文件：datasplit-hand.py
# 功能：划分手部静脉数据集（支持多源合并）
# 配置：src_roots包含多个源目录
python datasplit-hand.py
```

### 3. 静脉特征提取算法

#### **批量特征提取**
```bash
# 文件：algo-datagen-main.py
# 功能：批量提取静脉特征（支持三种算法）
# 算法选择：
#   1. max_curvature_extract - 最大曲率法
#   2. repeated_line_tracking - 重复线追踪法
#   3. gabor_vein_extract - Gabor滤波法
# 配置：修改INPUT_DIR和OUTPUT_DIR
# 切换算法：在代码中取消注释相应行
python algo-datagen-main.py
```

#### **单张图像特征提取测试**
```bash
# 文件：algo-datagen-single.py
# 功能：单张图像算法测试和调试
# 配置：修改input_file为测试图片路径
# 算法切换：在process_single_image函数中切换
python algo-datagen-single.py
```

### 4. LBP特征提取与匹配

#### **LBP特征系统**
```bash
# 文件：main-lbp.py
# 功能：LBP特征提取、匹配和评估
# 任务选择：修改TASK_ID
#   TASK_ID=1：单张图片可视化
#   TASK_ID=2：批量特征提取
#   TASK_ID=3：两张图片特征匹配
#   TASK_ID=4：整体数据集评估（ROC/EER）
# 配置：修改INPUT_DIR和OUTPUT_DIR路径
python main-lbp.py
```

### 5. 算法评估

```bash
# 文件：algo-val.py
# 功能：评估静脉提取算法的性能
# 配置：修改default_path为特征文件夹路径
# 或通过命令行参数：--path ./output/hand-rlt
python algo-val.py [--path 数据路径]
```


