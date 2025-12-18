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
