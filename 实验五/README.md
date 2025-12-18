# 一、实验目的
掌握使用PyTorch构建卷积神经网络（CNN）的方法。

实现对手写数字（MNIST）数据集的分类任务。

理解模型训练过程，包括前向传播、损失计算、反向传播与优化。

学习可视化模型性能与卷积特征图的方法。

# 二、实验环境
操作系统：Windows / Linux / macOS

编程语言：Python 3.8+

深度学习框架：PyTorch 1.10+

辅助库：torchvision, matplotlib, numpy

硬件设备：CPU / GPU（CUDA可用时自动使用）

# 三、实验内容
3.1 数据集准备
使用MNIST手写数字数据集，包含60,000张训练图像和10,000张测试图像。

图像大小为28×28，单通道灰度图。

通过torchvision.transforms将图像转换为Tensor格式。

使用DataLoader实现批量加载与随机打乱。

3.2 模型结构设计
构建一个包含两个卷积层和三个全连接层的CNN模型，结构如下：
<img width="776" height="635" alt="05c06c2e1d5bd977ce640606960502a0" src="https://github.com/user-attachments/assets/701c7fc1-129f-4ed0-be50-d36ea803e5c6" />
<img width="698" height="587" alt="1fe5243f833edf2c3eedff0d16b9be4c" src="https://github.com/user-attachments/assets/29069e7a-4d80-4d35-8613-ec8aefaafb9b" />
<img width="688" height="614" alt="202b80e30368856ae75ab10a51a38c04" src="https://github.com/user-attachments/assets/d30e3fe5-e5a4-43e9-a01a-a4c983f89415" />
<img width="767" height="639" alt="691e59ab7dbd4c30b54b6707b56bbca6" src="https://github.com/user-attachments/assets/9723cadd-9597-498e-9bea-a72238a510f9" />
<img width="735" height="413" alt="5227cc8873098ccf6b15c2d2e29a8982" src="https://github.com/user-attachments/assets/70259b1b-9b8d-4b99-9ac2-cb030bc54b19" />
<img width="752" height="657" alt="0778157dab9a7c50fb3db79e6c57e3fa" src="https://github.com/user-attachments/assets/2216ffe0-022a-4cda-8172-4e6d58ce4bf2" />

# 四、实验总结
这次实验的核心是用 PyTorch 构建 LeNet-5 网络，实现手写数字（0-9）识别。
搭建模型结构是第一个难点。LeNet-5 的结构描述虽清晰，但转化为代码时，必须保证各层张量维度严格匹配。定义 Conv2d 和 Linear 层时，输入输出通道数、特征图尺寸的计算是关键。最初我忽略了池化层对特征图尺寸的影响，错误计算了全连接层的输入维度，导致前向传播时出现维度不匹配错误。之后我借助 CNN 维度计算公式（输出尺寸 = floor ((输入尺寸 - 卷积核大小 + 2× 填充)/ 步长) + 1），并通过 print 语句逐层验证张量形状，比如确认 28×28 的 MNIST 图片经第一组卷积 - 池化后，维度变为 [batch_size, 6, 12, 12]，才解决了维度问题。
其次是优化器的选择对比。我分别用了 SGD 和 AdamW，相同学习率下，SGD 的损失曲线波动大、收敛慢；而学习率 0.01 的 AdamW 让训练更稳定，收敛也更快。为了量化效果并防范过拟合，我绘制了训练集与测试集的损失、准确率曲线。从 30 个 epoch 的曲线趋势能看出，训练与测试损失同步下降，测试准确率持续提升并趋于稳定。
最有趣的是卷积层可视化环节。我添加代码展示卷积层输出的特征图：第一层特征图能清晰捕捉数字的边缘、笔画（竖、横、斜笔），第二层则提取出拐角、环形等抽象特征，这让我直观理解了 “层次化特征提取” 的含义。
总的来说，这次实验将 LeNet-5 从理论设计落地为可运行的代码，还直观看到了网络的特征提取过程。整个实现过程，让我对卷积、池化、全连接等 CNN 基础操作的理解更为深刻。
