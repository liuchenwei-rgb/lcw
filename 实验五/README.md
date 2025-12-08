一、实验目的
掌握使用PyTorch构建卷积神经网络（CNN）的方法。

实现对手写数字（MNIST）数据集的分类任务。

理解模型训练过程，包括前向传播、损失计算、反向传播与优化。

学习可视化模型性能与卷积特征图的方法。

二、实验环境
操作系统：Windows / Linux / macOS

编程语言：Python 3.8+

深度学习框架：PyTorch 1.10+

辅助库：torchvision, matplotlib, numpy

硬件设备：CPU / GPU（CUDA可用时自动使用）

三、实验内容
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

四、实验总结
通过这次实验，我成功搭建了一个卷积神经网络，并用它来识别手写数字，这个神经网络能够比较准确地区分0-9这些数字图片，在训练时，我选择了AdamW优化器，学习率设为0.01。训练过程中，模型收敛得比较顺利，没有出现太大的波动，整体还算稳定。
通过绘制学习曲线和特征图，加深了对CNN工作机制的理解，有助于模型调试与优化，通过绘制准确率和损失的变化曲线，我能够直观地看到模型是怎么一步步变好的。特别是看到特征图可视化时，我能清楚看到：
第一层卷积主要提取的是数字的边缘、轮廓特征
第二层卷积提取的特征更抽象一些，可能是某些特定形状的组合


