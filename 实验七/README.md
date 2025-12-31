## 基于 Inception-v3/ResNet 的 Fashion-MNIST 图像分类实验报告
一、实验目的
掌握 PyTorch 框架实现图像分类任务的完整流程；
学习预训练模型（Inception-v3、ResNet）的迁移学习方法；
熟悉 Fashion-MNIST 数据集的处理与适配方法；
实现模型的训练、验证、评估及结果可视化；
分析深度学习模型在小数据集上的迁移学习性能。
二、实验原理
1. 核心模型
Inception-v3：Google 提出的卷积神经网络，通过多尺度 Inception 模块提取特征，结合批量归一化加速训练，原始输入尺寸为 299×299×3；
ResNet50：通过残差连接解决深层网络梯度消失问题，输入尺寸为 224×224×3，在图像分类任务中泛化能力较强。
2. 迁移学习
利用 ImageNet 预训练的模型权重，保留特征提取层，替换最后一层全连接层以适配 Fashion-MNIST 的 10 分类任务，通过微调参数适配新数据集。
3. Fashion-MNIST 数据集
包含 10 类时尚物品（T 恤、裤子、外套等）；
训练集 60000 张、测试集 10000 张，均为 28×28 单通道灰度图；
需通过通道扩展（转 3 通道）、尺寸缩放适配预训练模型输入。
三、实验环境
操作系统：Windows 10
深度学习框架：PyTorch 2.0.1
Python 版本：3.14.2
开发工具：PyCharm（虚拟环境）
硬件：CPU（或 NVIDIA GPU，支持 CUDA 加速）
四、实验步骤与代码实现
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from tqdm import tqdm

# ======================== 1. 环境配置与参数设置 ========================
# 设备配置：优先使用GPU，无则用CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 实验参数（可根据需求调整）
BATCH_SIZE = 32  # 批次大小（显存不足可调小）
LEARNING_RATE = 1e-4  # 学习率
EPOCHS = 10  # 训练轮数
DATA_RATIO = 0.3  # 使用数据集的比例（解决显存不足问题）
MODEL_TYPE = "resnet"  # 可选："inceptionv3" 或 "resnet"
SAVE_PATH = "./fashion_mnist_results"  # 结果保存路径

# 创建保存目录
os.makedirs(SAVE_PATH, exist_ok=True)

# ======================== 2. 数据预处理 ========================
# 针对Inception-v3/ResNet的输入要求处理数据：
# - Fashion-MNIST是28×28单通道灰度图，需转为3通道+缩放至对应尺寸
if MODEL_TYPE == "inceptionv3":
    input_size = 299  # Inception-v3要求299×299
else:
    input_size = 224  # ResNet要求224×224

transform = transforms.Compose([
    transforms.Resize(input_size),  # 缩放至模型要求尺寸
    transforms.Grayscale(num_output_channels=3),  # 单通道转3通道
    transforms.ToTensor(),  # 转为张量
    transforms.Normalize(  # 归一化（3通道对应3个均值/标准差）
        mean=[0.485, 0.456, 0.406],  # ImageNet预训练模型的归一化参数
        std=[0.229, 0.224, 0.225]
    )
])

# 加载Fashion-MNIST数据集
train_dataset = datasets.FashionMNIST(
    root='./data', train=True, download=True, transform=transform
)
test_dataset = datasets.FashionMNIST(
    root='./data', train=False, download=True, transform=transform
)

# 按比例采样数据（解决显存不足问题）
train_indices = np.random.choice(
    len(train_dataset), int(len(train_dataset) * DATA_RATIO), replace=False
)
test_indices = np.random.choice(
    len(test_dataset), int(len(test_dataset) * DATA_RATIO), replace=False
)
train_subset = Subset(train_dataset, train_indices)
test_subset = Subset(test_dataset, test_indices)

# 数据加载器（num_workers=0解决Windows多线程崩溃问题）
train_loader = DataLoader(
    train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)
test_loader = DataLoader(
    test_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)

# Fashion-MNIST类别名称
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# ======================== 3. 模型构建（迁移学习） ========================
def build_model(model_type):
    if model_type == "inceptionv3":
        # 加载预训练的Inception-v3
        model = models.inception_v3(pretrained=True)
        # 关闭辅助输出（解决分类计算错误问题）
        model.aux_logits = False
        # 替换最后一层全连接层（适配10分类）
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 10)

    elif model_type == "resnet":
        # 加载预训练的ResNet50
        model = models.resnet50(pretrained=True)
        # 替换最后一层全连接层
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 10)

    # 将模型移至设备
    model = model.to(device)
    return model


# 初始化模型
model = build_model(MODEL_TYPE)

# 损失函数与优化器（Adam解决SGD损失波动大问题）
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


# ======================== 4. 训练与验证函数 ========================
def train_model(model, train_loader, test_loader, criterion, optimizer, epochs):
    # 记录训练过程
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "params": {
            "batch_size": BATCH_SIZE,
            "lr": LEARNING_RATE,
            "epochs": epochs,
            "data_ratio": DATA_RATIO,
            "model_type": MODEL_TYPE
        }
    }

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 50)

        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        # 进度条
        train_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 反向传播与优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计损失和准确率
            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += torch.sum(preds == labels.data)
            train_total += labels.size(0)

            # 更新进度条
            train_bar.set_postfix(loss=loss.item(), acc=train_correct.item() / train_total)

        # 计算训练集平均损失和准确率
        epoch_train_loss = train_loss / len(train_loader.dataset)
        epoch_train_acc = train_correct.double() / len(train_loader.dataset)
        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc.item())

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():  # 禁用梯度计算加速验证
            val_bar = tqdm(test_loader, desc=f"Validating Epoch {epoch + 1}")
            for inputs, labels in val_bar:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels.data)
                val_total += labels.size(0)

                val_bar.set_postfix(loss=loss.item(), acc=val_correct.item() / val_total)

        # 计算验证集平均损失和准确率
        epoch_val_loss = val_loss / len(test_loader.dataset)
        epoch_val_acc = val_correct.double() / len(test_loader.dataset)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc.item())

        # 打印本轮结果
        print(f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f}")
        print(f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f}")

    return model, history


# ======================== 5. 执行训练 ========================
model, history = train_model(
    model, train_loader, test_loader, criterion, optimizer, EPOCHS
)

# ======================== 6. 结果保存与可视化 ========================
# 1. 保存模型参数
model_path = os.path.join(SAVE_PATH, f"{MODEL_TYPE}_fashion_mnist.pth")
torch.save(model.state_dict(), model_path)

# 2. 保存训练历史（方便复盘）
history_path = os.path.join(SAVE_PATH, f"{MODEL_TYPE}_history.json")
with open(history_path, 'w') as f:
    json.dump(history, f, indent=4)

# 3. 可视化训练过程
plt.figure(figsize=(12, 4))

# 损失曲线
plt.subplot(1, 2, 1)
plt.plot(history["train_loss"], label='Train Loss')
plt.plot(history["val_loss"], label='Val Loss')
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# 准确率曲线
plt.subplot(1, 2, 2)
plt.plot(history["train_acc"], label='Train Acc')
plt.plot(history["val_acc"], label='Val Acc')
plt.title('Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# 保存图片
plt.savefig(os.path.join(SAVE_PATH, f"{MODEL_TYPE}_training_curve.png"))
plt.show()

# 4. 打印最终结果
print("\n================ 实验结果 ================")
print(f"模型类型: {MODEL_TYPE}")
print(f"最终训练准确率: {history['train_acc'][-1]:.4f}")
print(f"最终验证准确率: {history['val_acc'][-1]:.4f}")
print(f"结果保存路径: {SAVE_PATH}")
五、实验结果
![ed7a47f40d125b47c1a92407a0a88a4e](https://github.com/user-attachments/assets/22a3bd43-0bda-4abc-95d8-1a564a53cc23)
![b75433d01ce662aee14c5ae30c7664b2](https://github.com/user-attachments/assets/741c3112-5e46-408e-af6e-07a431f5ea7a)
![5e5c4d69db457597a61dedb4d941f294](https://github.com/user-attachments/assets/3e921faf-9fc9-4b1e-b269-83306f657b98)
![b0df280287f7b91f1e3d7b282c1880f8](https://github.com/user-attachments/assets/971b0fa7-3a61-402f-8ece-5ad41687a234)
六、实验总结

