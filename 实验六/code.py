# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import Subset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子确保可复现性
torch.manual_seed(42)
np.random.seed(42)

# ---- 1. 参数设置 ----
batch_size = 16
epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# ---- 2. 数据准备 ----
# 定义数据转换：Fashion-MNIST 图片调整为299x299，灰度转3通道
transform = transforms.Compose([
    transforms.Resize(299),  # Inception-v3需要299x299输入
    transforms.Grayscale(num_output_channels=3),  # 将单通道转为3通道
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # 归一化
])

# 加载完整数据集
train_full = datasets.FashionMNIST(
    'data', 
    train=True, 
    download=True, 
    transform=transform
)
test_full = datasets.FashionMNIST(
    'data', 
    train=False, 
    download=True, 
    transform=transform
)

# 随机抽取数据子集（如果硬件资源有限）
n = 100  # 数据抽取比例，100表示使用100%的数据
n = min(max(n, 1), 100)  # 确保n在1-100之间

# 计算要抽取的样本数量
train_sample_size = len(train_full) * n // 100
test_sample_size = len(test_full) * n // 100

# 随机抽取索引
train_idx = np.random.choice(len(train_full), train_sample_size, replace=False)
test_idx = np.random.choice(len(test_full), test_sample_size, replace=False)

# 创建子集和数据加载器
train_dataset = Subset(train_full, train_idx)
test_dataset = Subset(test_full, test_idx)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2
)

print(f"训练集大小: {len(train_dataset)}")
print(f"测试集大小: {len(test_dataset)}")
print(f"使用 {n}% 的数据")

# ---- 3. 模型准备 ----
# 加载预训练的Inception-v3模型
model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)

# 修改全连接层，输出为10个类别（Fashion-MNIST有10类）
model.fc = nn.Linear(model.fc.in_features, 10)

# 如果是训练模式，需要处理辅助输出
model.aux_logits = False  # 禁用辅助输出以简化训练
model = model.to(device)

# 查看模型结构（可选）
# from torchsummary import summary
# summary(model, input_size=(3, 299, 299))

# 定义优化器和损失函数
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

# ---- 4. 训练与测试循环 ----
train_losses = []
test_losses = []
accuracies = []

for epoch in range(epochs):
    # 训练阶段
    model.train()
    train_loss = 0.0
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(x)  # 注意：这里不使用.logits属性
        
        # 计算损失
        loss = criterion(outputs, y)
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
        # 每100个batch打印一次进度
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}')
    
    # 计算平均训练损失
    avg_train_loss = train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    # 测试阶段
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            
            # 计算损失
            loss = criterion(outputs, y)
            test_loss += loss.item()
            
            # 计算准确率
            _, predicted = torch.max(outputs, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    
    # 计算平均测试损失和准确率
    avg_test_loss = test_loss / len(test_loader)
    accuracy = correct / total
    
    test_losses.append(avg_test_loss)
    accuracies.append(accuracy)
    
    # 打印每个epoch的结果
    print(f'Epoch {epoch+1}/{epochs}: '
          f'Train Loss: {avg_train_loss:.4f}, '
          f'Test Loss: {avg_test_loss:.4f}, '
          f'Accuracy: {accuracy:.4f}')
    print('-' * 60)

# ---- 5. 最终结果 ----
print(f'\n训练完成!')
print(f'最终测试准确率: {accuracies[-1]:.4f}')
print(f'最终测试损失: {test_losses[-1]:.4f}')

# ---- 6. 可视化结果 ----
plt.figure(figsize=(12, 4))

# 准确率曲线
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs+1), accuracies, 'b-o', linewidth=2, markersize=8)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Test Accuracy over Epochs')
plt.grid(True)
plt.xticks(range(1, epochs+1))

# 损失曲线
plt.subplot(1, 2, 2)
plt.plot(range(1, epochs+1), train_losses, 'r-^', label='Train Loss', linewidth=2, markersize=8)
plt.plot(range(1, epochs+1), test_losses, 'b-s', label='Test Loss', linewidth=2, markersize=8)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss')
plt.grid(True)
plt.legend()
plt.xticks(range(1, epochs+1))

plt.tight_layout()
plt.show()

# ---- 7. 保存模型 ----
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'accuracy': accuracies[-1],
    'epochs': epochs,
}, 'inception_v3_fashion_mnist.pth')

print("模型已保存为 'inception_v3_fashion_mnist.pth'")

# ---- 8. 类别标签（Fashion-MNIST） ----
class_names = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

print("\nFashion-MNIST类别标签:")
for i, name in enumerate(class_names):
    print(f"  {i}: {name}")
