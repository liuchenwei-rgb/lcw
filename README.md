# 实验三（3.1）：LeNet-5 网络的构建

## 一、实验目的


学会利用 PyTorch 设计 LeNet-5 网络结构，定义数据加载器、损失函数和优化器，构建完整的训练流程。以 MNIST 数据集为对象，利用 PyTorch 进行 LeNet-5 模型设计、数据加载、损失函数及优化器定义，评估模型的性能。

## **二、实验内容**

2.1 导入所需的依赖包

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
```

2.2 导入训练数据

```python
# 设置批处理大小
batch_size = 512

# 检查CUDA设备是否可用，并设置为设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义数据的转换方式，这里仅将数据转换为tensor
transform = transforms.Compose([transforms.ToTensor()])

# 加载训练集
train_loader = DataLoader(
    datasets.MNIST('data', train=True, download=True, transform=transform),
    batch_size=batch_size, shuffle=True
)

# 加载测试集
test_loader = DataLoader(
    datasets.MNIST('data', train=False, download=True, transform=transform),
    batch_size=batch_size, shuffle=True
)
```

2.3 定义模型

```python
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # 定义卷积层
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        # 定义全连接层
        self.fc1 = nn.Linear(5*5*16, 120)
        self.fc2 = nn.Linear(120, 84)
        # 定义分类层
        self.clf = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.sigmoid(x)
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        
        x = self.conv2(x)
        x = F.sigmoid(x)
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        
        x = torch.flatten(x, start_dim=1)
        
        x = self.fc1(x)
        x = F.sigmoid(x)
        
        x = self.fc2(x)
        x = F.sigmoid(x)
        
        x = self.clf(x)
        return x
```

2.4 进行模型初始化

```python
model = ConvNet().to(device)
optimizer = optim.AdamW(model.parameters(),lr = 1e-2)
model
```




    ConvNet(
      (conv1): Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
      (conv2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
      (fc1): Linear(in_features=400, out_features=120, bias=True)
      (fc2): Linear(in_features=120, out_features=84, bias=True)
      (clf): Linear(in_features=84, out_features=10, bias=True)
    )


2.5 对模型进行训练

```python
epochs = 30
accs, losses = [], []

for epoch in range(epochs):
    # 遍历训练集
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        # 计算损失并反向传播
        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()

    correct = 0
    testloss = 0
    # 在测试集上进行评估
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            x, y = x.to(device), y.to(device)
            out = model(x)
            # 计算测试集上的损失
            testloss += F.cross_entropy(out, y).item()
            # 计算准确率
            pred = out.max(dim=1, keepdim=True)[1]
            correct += pred.eq(y.view_as(pred)).sum().item()

    # 计算平均测试集损失和准确率
    acc = correct / len(test_loader.dataset)
    testloss /= (batch_idx + 1)
    accs.append(acc)
    losses.append(testloss)

    # 打印每个epoch的结果
    print('epoch: {}, loss: {:.4f}, acc: {:.4f}'.format(epoch, testloss, acc))
```

    epoch: 0, loss: 2.3015, acc: 0.1135
    epoch: 1, loss: 2.3017, acc: 0.1135
    epoch: 2, loss: 2.3010, acc: 0.1135
    epoch: 3, loss: 1.4345, acc: 0.4455
    epoch: 4, loss: 0.3515, acc: 0.8787
    epoch: 5, loss: 0.1555, acc: 0.9511
    epoch: 6, loss: 0.0917, acc: 0.9704
    epoch: 7, loss: 0.0812, acc: 0.9759
    epoch: 8, loss: 0.0717, acc: 0.9761
    epoch: 9, loss: 0.0583, acc: 0.9806
    epoch: 10, loss: 0.0600, acc: 0.9812
    epoch: 11, loss: 0.0550, acc: 0.9819
    epoch: 12, loss: 0.0522, acc: 0.9828
    epoch: 13, loss: 0.0510, acc: 0.9840
    epoch: 14, loss: 0.0498, acc: 0.9845
    epoch: 15, loss: 0.0459, acc: 0.9853
    epoch: 16, loss: 0.0496, acc: 0.9841
    epoch: 17, loss: 0.0431, acc: 0.9858
    epoch: 18, loss: 0.0402, acc: 0.9885
    epoch: 19, loss: 0.0390, acc: 0.9873
    epoch: 20, loss: 0.0412, acc: 0.9871
    epoch: 21, loss: 0.0419, acc: 0.9858
    epoch: 22, loss: 0.0529, acc: 0.9853
    epoch: 23, loss: 0.0424, acc: 0.9868
    epoch: 24, loss: 0.0454, acc: 0.9858
    epoch: 25, loss: 0.0395, acc: 0.9877
    epoch: 26, loss: 0.0418, acc: 0.9883
    epoch: 27, loss: 0.0424, acc: 0.9872
    epoch: 28, loss: 0.0391, acc: 0.9877
    epoch: 29, loss: 0.0380, acc: 0.9879


2.6 acc与loss变化趋势展示


```python
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)  # 一行两列，第一个子图
plt.plot(accs, label='Accuracy', color='blue')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Curve')
plt.legend()

plt.subplot(1, 2, 2)  # 一行两列，第二个子图
plt.plot(losses, label='Loss', color='red')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()

plt.tight_layout() 
plt.show()
```


​    
<img width="989" height="490" alt="image" src="https://github.com/user-attachments/assets/aba103ad-56fc-491b-a9c8-e8ea4b859e1f" />

​    

2.7 特征图可视化


```python
# 获取模型的卷积层的特征图
feature1 = model.conv1(x)
feature2 = model.conv2(feature1)
n = 5  

img = x.detach().cpu().numpy()[:n]
feature_map1 = feature1.detach().cpu().numpy()[:n]
feature_map2 = feature2.detach().cpu().numpy()[:n]

fig, ax = plt.subplots(3, n, figsize=(10, 10))
for i in range(n):
    # 对输入图像进行加和以便在灰度图中显示
    ax[0, i].imshow(img[i].sum(0), cmap='gray')
    ax[0, i].set_title(f'Input Image {i+1}')
    ax[1, i].imshow(feature_map1[i].sum(0), cmap='gray')
    ax[1, i].set_title(f'Feature Map 1.{i+1}')
    ax[2, i].imshow(feature_map2[i].sum(0), cmap='gray')
    ax[2, i].set_title(f'Feature Map 2.{i+1}')

# 调整子图间距
plt.tight_layout()
plt.show()
```


<img width="989" height="846" alt="image" src="https://github.com/user-attachments/assets/13defc46-5c76-4c73-ae23-78a7f52aa484" />


# 三、实验小结
​本次实验基于PyTorch框架，成功设计并实现了经典的LeNet-5卷积神经网络，并在MNIST手写数字数据集上完成了模型的训练与评估。

实验首先完成了数据预处理流程的搭建，定义了数据转换方法并配置了数据加载器。在模型构建阶段，通过自定义ConvNet类，精确地复现了LeNet-5的网络架构，该架构包含两个卷积-池化层对以及后续的全连接层。

在模型训练中，选用AdamW优化器与交叉熵损失函数，经过30个训练周期后，模型在测试集上的准确率显著提高，同时训练损失持续下降，表明模型具备了有效的学习能力。训练过程中的准确率与损失变化曲线进一步直观地验证了模型收敛的稳定性与有效性。

此外，实验还通过可视化技术，展示了模型在卷积层所提取的特征图。这一分析加深了对卷积神经网络层次化特征抽象过程的理解，揭示了其从原始像素中逐步学习形状和轮廓等本质特征的内在机制。

综上所述，本实验不仅成功复现了LeNet-5模型并取得了良好的分类性能，更通过全流程的实践，强化了关于深度学习模型设计、训练、评估及内部机理分析的综合实践能力。
