# 实验三（3.1）：LeNet-5 网络的构建

## 一、实验目的


学会利用 PyTorch 设计 LeNet-5 网络结构，定义数据加载器、损失函数和优化器，构建完整的训练流程。以 MNIST 数据集为对象，利用 PyTorch 进行 LeNet-5 模型设计、数据加载、损失函数及优化器定义，评估模型的性能。

## **二、实验内容**
```
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像并转换为灰度图
img_a = cv2.imread("a.jpg")
img_b = cv2.imread("b.jpg")
gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)

# 初始化SIFT检测器并提取特征
sift = cv2.SIFT_create()
kp_a, des_a = sift.detectAndCompute(gray_a, None)  # 图像a的关键点和描述符
kp_b, des_b = sift.detectAndCompute(gray_b, None)  # 图像b的关键点和描述符

# 使用FLANN匹配器进行特征匹配
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)  # 检查次数越多，匹配越准确但速度越慢
flann = cv2.FlannBasedMatcher(index_params, search_params)

# k=2表示每个特征点返回2个最佳匹配
matches = flann.knnMatch(des_a, des_b, k=2)

# 应用Lowe's比率测试筛选优质匹配点
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:  # 比率阈值通常取0.7-0.8
        good_matches.append(m)

# 绘制匹配的SIFT关键点
matched_keypoints_img = cv2.drawMatches(
    img_a, kp_a, img_b, kp_b, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

# 提取匹配点的坐标
src_pts = np.float32([kp_b[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)  # 图像b的关键点
dst_pts = np.float32([kp_a[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)  # 图像a的关键点

# 使用RANSAC算法估计单应矩阵(透视变换矩阵)
H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# 获取输入图像尺寸
h_a, w_a = img_a.shape[:2]
h_b, w_b = img_b.shape[:2]

# 计算图像b变换后的四个角点坐标
pts = np.float32([[0, 0], [0, h_b], [w_b, h_b], [w_b, 0]]).reshape(-1, 1, 2)
dst_corners = cv2.perspectiveTransform(pts, H)

# 确定拼接后图像的最终尺寸(包含所有像素)
all_corners = np.concatenate([
    dst_corners,
    np.float32([[0, 0], [w_a, 0], [w_a, h_a], [0, h_a]]).reshape(-1, 1, 2)
], axis=0)
[x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
[x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

# 创建平移矩阵，确保所有像素都在可见区域内
translation_matrix = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float32)

# 对图像b进行透视变换和平移
fus_img = cv2.warpPerspective(
    img_b,
    translation_matrix @ H,  # 组合平移矩阵和单应矩阵
    (x_max - x_min, y_max - y_min)  # 输出图像尺寸
```
<img width="2409" height="1075" alt="0f0031c0eab952777317191082a600e3" src="https://github.com/user-attachments/assets/aeb47f58-3e7b-40a2-8ba9-4fa0bee7cb73" />

# 三、实验小结
​本次实验基于PyTorch框架，成功设计并实现了经典的LeNet-5卷积神经网络，并在MNIST手写数字数据集上完成了模型的训练与评估。

实验首先完成了数据预处理流程的搭建，定义了数据转换方法并配置了数据加载器。在模型构建阶段，通过自定义ConvNet类，精确地复现了LeNet-5的网络架构，该架构包含两个卷积-池化层对以及后续的全连接层。

在模型训练中，选用AdamW优化器与交叉熵损失函数，经过30个训练周期后，模型在测试集上的准确率显著提高，同时训练损失持续下降，表明模型具备了有效的学习能力。训练过程中的准确率与损失变化曲线进一步直观地验证了模型收敛的稳定性与有效性。

此外，实验还通过可视化技术，展示了模型在卷积层所提取的特征图。这一分析加深了对卷积神经网络层次化特征抽象过程的理解，揭示了其从原始像素中逐步学习形状和轮廓等本质特征的内在机制。

综上所述，本实验不仅成功复现了LeNet-5模型并取得了良好的分类性能，更通过全流程的实践，强化了关于深度学习模型设计、训练、评估及内部机理分析的综合实践能力。
