# 实验三 基于SIFT特征和RANSAC算法的图像拼接实验报告
## 一、实验目的
学习SIFT（尺度不变特征变换）特征检测与描述的原理
掌握基于特征点的图像匹配技术
理解RANSAC（随机抽样一致）算法在图像配准中的应用
实现两幅图像的特征匹配、单应矩阵估计和透视变换拼接

## 二、实验原理
1. SIFT特征提取
SIFT（Scale-Invariant Feature Transform）算法具有尺度、旋转和光照不变性。其主要步骤包括：
尺度空间极值检测
关键点精确定位
方向分配
关键点描述符生成
2. FLANN特征匹配
FLANN（Fast Library for Approximate Nearest Neighbors）是基于KD树的快速近似最近邻搜索算法，比暴力匹配（Brute-Force）具有更高的效率。
3. Lowe's比率测试
用于筛选优质匹配点：
距离比值 = 最近邻距离 / 次近邻距离
当比值小于阈值（通常0.7-0.8）时保留该匹配
4. RANSAC与单应矩阵
RANSAC用于从包含异常值的匹配点中鲁棒地估计单应矩阵H（3×3透视变换矩阵）：
H = [[h11, h12, h13],
     [h21, h22, h23],
     [h31, h32, 1]]

## 三、实验环境
Python 3.8+
OpenCV 4.5+
NumPy
Matplotlib

## 四、实验内容
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

## 五、实验小结
这次实验的核心任务是用PyTorch框架，从头开始搭建一个LeNet-5网络，并且让它学会了准确识别0-9的手写数字图片。

第一个比较花时间的地方是模型结构。LeNet-5的结构描述虽然清晰，但将其转换为精确的PyTorch代码，关键在于确保各层间张量维度的严格匹配。例如，在定义Conv2d和Linear层时，输入/输出通道数与特征图尺寸的准确计算是首要挑战。首次实现时，我忽略了池化层对特征图尺寸的影响，错误计算了展平后的维度，导致模型在前向传播时因维度不匹配而抛出错误。于是我回归到CNN维度计算的基本公式：输出尺寸 = floor((输入尺寸 - 卷积核大小 + 2 * 填充) / 步长) + 1。通过插入print(x.shape)语句，在模型调试阶段逐层验证了张量的实际尺寸，并与理论计算结果进行比对。例如，验证了MNIST图片（1×28×28）经过第一组卷积-池化后是否准确地变为[batch_size, 6, 12, 12]。

第二个是优化器的选择。实验中，我对比了SGD与AdamW两种优化器。观察到在相同学习率下，SGD的损失函数下降曲线波动较大，收敛速度较慢；而采用AdamW优化器（学习率设为0.01）后，训练过程显著更稳定，收敛也更快。
为了量化这一观察并防止过拟合，我同时绘制了训练集和测试集在每个epoch后的损失与准确率曲线。关键判别依据是两条损失曲线的趋势：在30个epoch的训练中，训练损失与测试损失始终保持同步下降的趋势，且测试准确率稳步提升，最终稳定在一个较高水平

最让我觉得有意思的，是最后看模型世界。 我添加了可视化卷积层输出的代码。当看到第一层卷积出来的那些“特征图”时，感觉特别直观——它们就像是用各种角度的笔画，在描数字的边缘和轮廓。有些图能明显看出数字的竖笔，有些则专门抓横笔或者斜笔。而第二层卷积出来的图就抽象多了，比如一个拐角，或者一个小的环形。这让我一下子理解了“层次化特征提取”是什么意思

总的来说，这次实验让我把一个经典的CNN结构从图纸变成了真正能跑的代码，并且亲眼看到了它是怎么一层层学会识别数字的。实现LeNet-5结构的整个过程，让我对卷积、池化、全连接这些基本操作的理解。
