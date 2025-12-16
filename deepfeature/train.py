import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import datetime
import sys
import os
from torch.utils.tensorboard import SummaryWriter

# 引入自定义模块
from model import VeinNet, VeinNetV2
from utils import Logger, get_device

# 获取设备
device = get_device()
print(f"Is CUDA available: {torch.cuda.is_available()}")

# 全局配置
INPUT_SIZE = (64, 64)
DATATYPE = "hand"
BATCH_SIZE = 2
LEARNING_RATE = 0.01
NUM_EPOCHS = 600
NUM_CLASSES = 15  # 根据你的代码逻辑输出层是15

# 数据预处理与增强
train_transform = transforms.Compose(
    [
        transforms.Resize(INPUT_SIZE),
        transforms.ColorJitter(
            brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05
        ),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ToTensor(),
    ]
)

test_transform = transforms.Compose(
    [transforms.Resize(INPUT_SIZE), transforms.ToTensor()]
)


def load_data():
    """加载数据集"""
    train_path = f"./output/{DATATYPE}data/train"
    test_path = f"./output/{DATATYPE}data/test"
    val_path = f"./output/{DATATYPE}data/val"

    # 检查路径是否存在，避免报错
    if not os.path.exists(train_path):
        print(f"Error: Data path {train_path} not found.")
        sys.exit(1)

    train_data = torchvision.datasets.ImageFolder(
        root=train_path, transform=train_transform
    )
    test_data = torchvision.datasets.ImageFolder(
        root=test_path, transform=test_transform
    )
    val_data = torchvision.datasets.ImageFolder(root=val_path, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=BATCH_SIZE, shuffle=True
    )

    return train_loader, val_loader, test_loader


def main():
    print("Start Training...")
    net = VeinNet(num_classes=NUM_CLASSES).to(device)
    netname = net.__class__.__name__
    # 1. 设置 TensorBoard 和 Logger
    log_dir = f"runs/{netname}_{DATATYPE}_{INPUT_SIZE}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir)
    log_file_path = os.path.join(log_dir, "train_log.txt")

    # 保存原始 stdout 以便恢复
    original_stdout = sys.stdout
    sys.stdout = Logger(log_file_path)

    print(f"Log file created at: {log_file_path}")
    print(f"Using Device: {device}")
    print("Input Size:", INPUT_SIZE)
    print("Type:", DATATYPE)

    # 2. 加载数据
    train_loader, val_loader, test_loader = load_data()

    # 3. 实例化模型
    print("model:", netname)
    # 损失函数与优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE)

    # 4. 可视化模型结构
    dummy_input = torch.rand(1, 3, INPUT_SIZE[0], INPUT_SIZE[1]).to(device)
    writer.add_graph(net, dummy_input)

    best_acc = 0.0
    save_path = f"./output/best_{netname}_{DATATYPE}_{INPUT_SIZE}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"

    # 确保保存路径存在
    os.makedirs("./output", exist_ok=True)

    for epoch in range(NUM_EPOCHS):
        # ==================== 训练阶段 ====================
        net.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)  # 此时调用 forward，返回分类结果
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_acc = correct / total

        # ==================== 验证阶段 ====================
        net.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0

        with torch.no_grad():
            for data in val_loader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_acc = val_correct / val_total

        print(
            f"Epoch {epoch+1}: Train Loss: {epoch_train_loss:.4f} | Val Acc: {epoch_val_acc:.2%}"
        )

        writer.add_scalars(
            "Loss", {"Train": epoch_train_loss, "Validation": epoch_val_loss}, epoch
        )
        writer.add_scalars(
            "Accuracy", {"Train": epoch_train_acc, "Validation": epoch_val_acc}, epoch
        )

        # 保存最佳模型
        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            torch.save(net.state_dict(), save_path)
            print(f"Best model saved to {save_path}")

    writer.close()
    print("Done Training!")
    print(f"Best Validation Accuracy: {best_acc:.2%}")

    # ==================== 测试阶段 ====================
    print("Starting Testing with Best Model...")
    net.load_state_dict(torch.load(save_path))
    net.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    print(f"Final Test Accuracy: {test_correct / test_total:.2%}")

    # 恢复标准输出
    sys.stdout = original_stdout


if __name__ == "__main__":
    main()
