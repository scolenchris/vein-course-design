import torch
import torch.nn as nn


class VeinNet(nn.Module):
    # 这个模型参数量很少，运算量很小，是为了方便没有GPU的同学做实验
    def __init__(self, num_classes=15):
        super(VeinNet, self).__init__()
        # 卷积层
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=5, padding=0)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=0)
        self.conv3 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, padding=0
        )
        self.conv4 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=0
        )

        # Batch Normalization
        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(16)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(64)

        # 池化与激活
        self.pool = nn.MaxPool2d(2, 2)
        self.act = nn.LeakyReLU()

        # 特征处理层
        self.feature = nn.AdaptiveAvgPool2d(1)
        # 分类层
        self.x2c = nn.Linear(64, num_classes)

    def forward(self, x):
        # 复用 extract_feature 流程，保持训练和推理一致
        x = self.extract_feature(x)
        c = self.x2c(x)
        return c

    # 【新增】专门用于任务二：提取静脉特征（64维向量）
    def extract_feature(self, x):
        # 第一层
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.pool(x)
        x = self.act(x)
        # 第二层
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.pool(x)
        x = self.act(x)
        # 第三层
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.pool(x)
        x = self.act(x)
        # 第四层
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.pool(x)
        x = self.act(x)

        # 输出特征 (这里对应你提到的 x = self.feature(x).view(-1, 64))
        x = self.feature(x).view(x.size(0), -1)
        return x


class VeinNetV2(nn.Module):
    def __init__(self, num_classes=15):
        super(VeinNetV2, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2, 2)
        self.act = nn.LeakyReLU(0.1)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.extract_feature(x)
        x = self.fc(x)
        return x

    def extract_feature(self, x):
        x = self.pool(self.act(self.bn1(self.conv1(x))))
        x = self.pool(self.act(self.bn2(self.conv2(x))))
        x = self.pool(self.act(self.bn3(self.conv3(x))))
        x = self.pool(self.act(self.bn4(self.conv4(x))))
        x = self.pool(self.act(self.bn5(self.conv5(x))))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return x
