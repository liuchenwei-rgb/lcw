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

