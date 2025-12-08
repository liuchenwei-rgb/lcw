import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# 配置参数
batch_size = 512
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epochs = 30

# 数据加载
trainloader = DataLoader(
    datasets.MNIST('data', train=True, download=True,
                   transform=transforms.Compose([transforms.ToTensor()])),
    batch_size=batch_size, shuffle=True
)

testloader = DataLoader(
    datasets.MNIST('data', train=False, download=True,
                   transform=transforms.Compose([transforms.ToTensor()])),
    batch_size=batch_size, shuffle=True
)

# 定义 LeNet 网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 卷积层定义
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        # 全连接层定义
        self.fc1 = nn.Linear(in_features=4*4*16, out_features=120)  # 修正：应该是4*4*16
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.clf = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        # 第一层卷积 + sigmoid激活 + 平均池化
        x = self.conv1(x)
        x = F.sigmoid(x)
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        
        # 第二层卷积 + sigmoid激活 + 平均池化
        x = self.conv2(x)
        x = F.sigmoid(x)
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = F.sigmoid(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = self.clf(x)
        
        return x

# 初始化模型、优化器
model = Net().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-2)

# 训练与测试循环
train_losses = []
test_accs = []
test_losses = []

for epoch in range(epochs):
    # 训练阶段
    model.train()
    train_loss = 0
    for batch_idx, (x, y) in enumerate(trainloader):
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        out = model(x)
        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    # 测试阶段
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            test_loss += F.cross_entropy(out, y).item()
            
            _, predicted = torch.max(out, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    
    # 计算平均损失和准确率
    avg_train_loss = train_loss / len(trainloader)
    avg_test_loss = test_loss / len(testloader)
    accuracy = correct / total
    
    # 保存结果
    train_losses.append(avg_train_loss)
    test_losses.append(avg_test_loss)
    test_accs.append(accuracy)
    
    # 打印每轮结果
    print(f'Epoch {epoch+1}/{epochs}: '
          f'Train Loss: {avg_train_loss:.4f}, '
          f'Test Loss: {avg_test_loss:.4f}, '
          f'Accuracy: {accuracy:.4f}')

print(f'\n训练完成！最终测试准确率: {test_accs[-1]:.4f}')

# 可视化结果
plt.figure(figsize=(12, 4))

# 准确率曲线
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs+1), test_accs, 'b-', label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Test Accuracy Curve')
plt.grid(True)
plt.legend()

# 损失曲线
plt.subplot(1, 2, 2)
plt.plot(range(1, epochs+1), train_losses, 'r-', label='Train Loss')
plt.plot(range(1, epochs+1), test_losses, 'b-', label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curves')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# 保存模型
torch.save(model.state_dict(), 'lenet_mnist.pth')
print("模型已保存为 'lenet_mnist.pth'")
