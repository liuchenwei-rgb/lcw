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
这次实验的核心任务是用PyTorch框架，从头开始搭建一个LeNet-5网络，并且让它学会了准确识别0-9的手写数字图片。

第一个比较花时间的地方是模型结构。LeNet-5的结构描述虽然清晰，但将其转换为精确的PyTorch代码，关键在于确保各层间张量维度的严格匹配。例如，在定义Conv2d和Linear层时，输入/输出通道数与特征图尺寸的准确计算是首要挑战。首次实现时，我忽略了池化层对特征图尺寸的影响，错误计算了展平后的维度，导致模型在前向传播时因维度不匹配而抛出错误。于是我回归到CNN维度计算的基本公式：输出尺寸 = floor((输入尺寸 - 卷积核大小 + 2 * 填充) / 步长) + 1。通过插入print(x.shape)语句，在模型调试阶段逐层验证了张量的实际尺寸，并与理论计算结果进行比对。例如，验证了MNIST图片（1×28×28）经过第一组卷积-池化后是否准确地变为[batch_size, 6, 12, 12]。

第二个是优化器的选择。实验中，我对比了SGD与AdamW两种优化器。观察到在相同学习率下，SGD的损失函数下降曲线波动较大，收敛速度较慢；而采用AdamW优化器（学习率设为0.01）后，训练过程显著更稳定，收敛也更快。 为了量化这一观察并防止过拟合，我同时绘制了训练集和测试集在每个epoch后的损失与准确率曲线。关键判别依据是两条损失曲线的趋势：在30个epoch的训练中，训练损失与测试损失始终保持同步下降的趋势，且测试准确率稳步提升，最终稳定在一个较高水平

最让我觉得有意思的，是最后看模型世界。 我添加了可视化卷积层输出的代码。当看到第一层卷积出来的那些“特征图”时，感觉特别直观——它们就像是用各种角度的笔画，在描数字的边缘和轮廓。有些图能明显看出数字的竖笔，有些则专门抓横笔或者斜笔。而第二层卷积出来的图就抽象多了，比如一个拐角，或者一个小的环形。这让我一下子理解了“层次化特征提取”是什么意思

总的来说，这次实验让我把一个经典的CNN结构从图纸变成了真正能跑的代码，并且亲眼看到了它是怎么一层层学会识别数字的。实现LeNet-5结构的整个过程，让我对卷积、池化、全连接这些基本操作的理解
