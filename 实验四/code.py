# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

# 配置参数
batch_size = 2048
device = torch.device('cpu')  # 可改为 'cuda' 启用GPU加速

# 加载MNIST数据集
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        root='data',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor()
        ])),
    batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        root='data',
        train=False,
        transform=transforms.Compose([
            transforms.ToTensor()
        ])),
    batch_size=batch_size, shuffle=True)

# 查看训练集数据形状
print(train_loader.dataset.data.shape)

# 定义MLP模型
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.l1 = nn.Linear(in_features=784, out_features=128)  # 输入层到隐藏层
        self.l2 = nn.Linear(in_features=128, out_features=10)  # 隐藏层到输出层

    def forward(self, x):
        a1 = self.l1(x)
        x1 = F.relu(a1)  # ReLU激活函数
        a2 = self.l2(x1)
        x2 = a2  # 输出层不使用激活 (cross_entropy已包含softmax)
        return x2

# 初始化模型、优化器
model = MLP().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.1)
epochs = 10

# 训练与测试循环
for epoch in range(epochs):
    # 训练阶段
    model.train()
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.view(x.shape[0], -1).to(device), y.to(device)  # 展平图像(28 * 28=784)
        output = model(x)
        optimizer.zero_grad()  # 清空梯度
        loss = F.cross_entropy(output, y)  # 计算交叉熵损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

    # 测试阶段
    model.eval()
    correct = 0
    test_loss = 0
    
    with torch.no_grad():  # 禁用梯度计算
        for batch_idx, (x, y) in enumerate(test_loader):
            x, y = x.view(x.shape[0], -1).to(device), y.to(device)
            output = model(x)
            test_loss += F.cross_entropy(output, y).item()  # 累加测试损失
            pred = output.max(1, keepdim=True)[1]  # 预测类别
            correct += pred.eq(y.view_as(pred)).sum().item()  # 统计正确个数
    
    # 计算平均测试损失和准确率
    test_loss = test_loss / (batch_idx + 1)
    acc = correct / len(test_loader.dataset)
    print(f'epoch: {epoch}, loss: {test_loss:.4f}, acc: {acc:.4f}')
