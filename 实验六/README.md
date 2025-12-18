# 实验六 基于Inception-v3的Fashion-MNIST图像分类实验报告
## 一、实验目的
掌握使用PyTorch框架进行图像分类任务的基本流程
学习使用预训练模型（Inception-v3）进行迁移学习
熟悉Fashion-MNIST数据集的特点和处理方法
实现完整的模型训练、验证和评估流程
分析深度学习模型在图像分类任务中的性能表现
## 二、实验原理
1. Inception-v3模型
Inception-v3是Google提出的卷积神经网络架构，具有以下特点：
使用Inception模块实现多尺度特征提取
引入批量归一化（Batch Normalization）加速训练
采用标签平滑（Label Smoothing）正则化
原始输入尺寸为299×299×3
2. 迁移学习
通过使用在ImageNet数据集上预训练的Inception-v3模型，迁移到Fashion-MNIST分类任务：
保留预训练的特征提取层
替换最后的全连接层以适应10分类任务
微调模型参数以适应新数据集
3. Fashion-MNIST数据集
包含10个类别的时尚物品图像
训练集：60,000张28×28灰度图像
测试集：10,000张28×28灰度图像
类别：T恤、裤子、套衫、裙子、外套、凉鞋、衬衫、运动鞋、包、踝靴
## 三、实验环境
操作系统：不限（支持CUDA的GPU环境可加速训练）
深度学习框架：PyTorch 1.8+
Python版本：3.6+
依赖库：
torch
torchvision
numpy
matplotlib（用于可视化）
## 四、实验步骤与代码实现

<img width="1092" height="992" alt="image" src="https://github.com/user-attachments/assets/192560e9-3abb-481f-85a8-e56db527fc6d" />


<img width="781" height="821" alt="image" src="https://github.com/user-attachments/assets/7ab697fc-a146-411b-9167-4e89bf05a6d1" />

## 五、实验结果

<img width="1200" height="400" alt="50de9967ab218f23362b2b41964a320b" src="https://github.com/user-attachments/assets/a418ca7a-65b4-4d2f-867a-736930ab0e85" />

## 六、实验总结

本次实验用 PyTorch 框架和预训练的 Inception-v3 模型做 Fashion-MNIST 服装分类任务，过程中主要解决了几个基础问题，确保模型能正常训练并完成分类。

首先是数据和模型输入不匹配的问题，Inception-v3 需要 3 通道、299×299 尺寸的图像，而 Fashion-MNIST 是单通道、28×28 的灰度图，直接使用会报错，解决办法是把灰度图转成 3 通道（添加Grayscale(num_output_channels=3)），将图像放大到 299×299（用Resize(299)），同时把归一化参数从 1 个值调整为 3 个值，保证和通道数对应。

然后是模型分类类别不符的问题，Inception-v3 原本用于 1000 类分类，而 Fashion-MNIST 只有 10 类服装，直接使用会出现类别数不匹配，还会因模型 “辅助输出” 导致计算错误，解决时把模型最后一层改成输出 10 类，同时关掉 “辅助输出” 功能（设置model.aux_logits = False），让模型只输出一个分类结果。训练过程中也遇到了故障，一是 Windows 系统下用多线程加载数据会崩溃，改成单线程（num_workers=0）后恢复正常；二是全量数据训练时显存不够，且用 SGD 优化器训练时损失波动大，解决办法是只取部分数据（比如 30%）训练，同时将优化器换成 Adam，之后损失下降更平稳，收敛速度也变快了。

另外，最初只保存模型参数，没记录训练过程的损失和准确率，后续想分析训练效果时没有数据，所以后来保存模型时，顺便把每轮的损失、准确率以及使用的数据比例、训练轮数都存了下来，方便后续复盘。最终通过这些调整，模型能正常训练，
