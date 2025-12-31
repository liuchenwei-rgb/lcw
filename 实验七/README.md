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

"""
Fashion-MNIST图像分类实验（Inception-v3/ResNet迁移学习）
适配GitHub直接复制运行，解决输入适配、显存、训练稳定性等问题
"""
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

设备配置：优先GPU，无则CPU

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

实验核心参数

    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    EPOCHS = 10
    DATA_RATIO = 0.3  # 数据集采样比例（缓解显存压力）
    MODEL_TYPE = "resnet"  # 可选："inceptionv3" / "resnet"
    SAVE_PATH = "./fashion_mnist_results"

创建结果保存目录

    os.makedirs(SAVE_PATH, exist_ok=True)

数据预处理：适配预训练模型输入要求

    input_size = 299 if MODEL_TYPE == "inceptionv3" else 224
    transform = transforms.Compose([

    transforms.Resize(input_size),
    transforms.Grayscale(num_output_channels=3),  # 单通道转3通道
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

加载Fashion-MNIST数据集
train_dataset = datasets.FashionMNIST(
    root='./data', train=True, download=True, transform=transform
)
test_dataset = datasets.FashionMNIST(
    root='./data', train=False, download=True, transform=transform
)

数据集采样（减少显存占用）
train_indices = np.random.choice(
    len(train_dataset), int(len(train_dataset)*DATA_RATIO), replace=False
)
test_indices = np.random.choice(
    len(test_dataset), int(len(test_dataset)*DATA_RATIO), replace=False
)
train_subset = Subset(train_dataset, train_indices)
test_subset = Subset(test_dataset, test_indices)

数据加载器（Windows下num_workers=0避免多线程崩溃）
train_loader = DataLoader(
    train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)
test_loader = DataLoader(
    test_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)

类别名称映射
class_names = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

构建迁移学习模型
def build_model(model_type):
    if model_type == "inceptionv3":
        model = models.inception_v3(pretrained=True)
        model.aux_logits = False  # 关闭辅助输出
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 10)  # 适配10分类
    elif model_type == "resnet":
        model = models.resnet50(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 10)  # 适配10分类
    model = model.to(device)
    return model

初始化模型、损失函数、优化器
model = build_model(MODEL_TYPE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

训练与验证函数
def train_model(model, train_loader, test_loader, criterion, optimizer, epochs):
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "params": {
            "batch_size": BATCH_SIZE, "lr": LEARNING_RATE,
            "epochs": epochs, "data_ratio": DATA_RATIO, "model_type": MODEL_TYPE
        }
    }
        
        for epoch in range(epochs): 
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 50)

        # 训练阶段
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        train_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += torch.sum(preds == labels.data)
            train_total += labels.size(0)
            
            train_bar.set_postfix(loss=loss.item(), acc=train_correct/train_total)

        # 训练集指标计算
        epoch_train_loss = train_loss / len(train_loader.dataset)
        epoch_train_acc = train_correct.double() / len(train_loader.dataset)
        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc.item())

        # 验证阶段
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        
        with torch.no_grad():
            val_bar = tqdm(test_loader, desc=f"Validating Epoch {epoch+1}")
            for inputs, labels in val_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels.data)
                val_total += labels.size(0)
                
                val_bar.set_postfix(loss=loss.item(), acc=val_correct/val_total)

        # 验证集指标计算
        epoch_val_loss = val_loss / len(test_loader.dataset)
        epoch_val_acc = val_correct.double() / len(test_loader.dataset)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc.item())

        # 打印本轮结果
        print(f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f}")
        print(f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f}")

    return model, history

执行训练
model, history = train_model(
    model, train_loader, test_loader, criterion, optimizer, EPOCHS
)

保存模型与训练历史
torch.save(model.state_dict(), os.path.join(SAVE_PATH, f"{MODEL_TYPE}_fashion_mnist.pth"))
with open(os.path.join(SAVE_PATH, f"{MODEL_TYPE}_history.json"), 'w') as f:
    json.dump(history, f, indent=4)

可视化训练曲线
plt.figure(figsize=(12, 4))

损失曲线
plt.subplot(1, 2, 1)
plt.plot(history["train_loss"], label='Train Loss')
plt.plot(history["val_loss"], label='Val Loss')
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

准确率曲线
plt.subplot(1, 2, 2)
plt.plot(history["train_acc"], label='Train Acc')
plt.plot(history["val_acc"], label='Val Acc')
plt.title('Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

保存可视化结果
plt.savefig(os.path.join(SAVE_PATH, f"{MODEL_TYPE}_training_curve.png"))
plt.show()

打印最终结果
print("\n================ Experiment Results ================")
print(f"Model Type: {MODEL_TYPE}")
print(f"Final Train Accuracy: {history['train_acc'][-1]:.4f}")
print(f"Final Validation Accuracy: {history['val_acc'][-1]:.4f}")
print(f"Results Saved to: {SAVE_PATH}")


五、实验结果
![ed7a47f40d125b47c1a92407a0a88a4e](https://github.com/user-attachments/assets/22a3bd43-0bda-4abc-95d8-1a564a53cc23)
![b75433d01ce662aee14c5ae30c7664b2](https://github.com/user-attachments/assets/741c3112-5e46-408e-af6e-07a431f5ea7a)
![5e5c4d69db457597a61dedb4d941f294](https://github.com/user-attachments/assets/3e921faf-9fc9-4b1e-b269-83306f657b98)
![b0df280287f7b91f1e3d7b282c1880f8](https://github.com/user-attachments/assets/971b0fa7-3a61-402f-8ece-5ad41687a234)

六、实验总结

本次实验用PyTorch搭了Inception-v3的迁移学习，跑了Fashion-MNIST的分类，主要是出现几个问题，让模型能好好训练出结果。

先是数据和模型Inception-v3要的是3通道299×299图，Fashion-MNIST却是单通道28×28的灰度图，直接用就报错。后来给灰度图补了2个通道（加Grayscale(num_output_channels=3)），再把图放大到299×299（用Resize(299)），归一化参数也改成3个，跟通道数对上，这问题就解决了。

然后是模型分类数不对，原模型是分1000类的，Fashion-MNIST就10类，还带个“辅助输出”搅和计算。于是把最后一层改成输出10类，再把aux_logits关了（设成False），模型就只出一个分类结果了。

训练时也踩了坑：Windows用多线程读数据直接崩，改成单线程（num_workers=0）就好了；全量数据训的话显存不够，SGD优化器还让损失忽高忽低，后来只取30%数据训，换成Adam优化器，损失降得稳多了，收敛也快。

最开始只存模型参数，想复盘训练效果没数据，后面就把每轮的损失、准确率，还有用了多少数据、训了几轮都存在文件里，后面看结果也方便。

这么调整下来，模型顺顺当当训完了，分类也能跑出结果，迁移学习在小数据集上的适配也算摸清楚了基本路子。别那么口语化

