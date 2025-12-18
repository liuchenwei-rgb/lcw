# -*- coding: utf-8 -*-
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.util import random_noise
from typing import Tuple, Optional

# 1. 读取图像
img = cv2.imread("image.png")  # 请确保当前目录下存在 image.png
if img is None:
    raise FileNotFoundError("未找到图片 'image.png'，请检查文件路径。")

# 2. 访问像素值 (OpenCV 使用 BGR 颜色顺序)
b, g, r = img[100, 100]
print(f"位于(100,100)的像素值 - B(Blue): {b}, G(Green): {g}, R(Red): {r}")
print("注意：OpenCV默认使用BGR颜色空间，而非RGB。")

# 3. 直接显示BGR图像（颜色会异常）
plt.imshow(img)
plt.title("直接显示BGR图像 (颜色异常)")
plt.axis('off')
plt.show()

# 4. 转换为RGB并显示
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(rgb_img)
plt.title("转换为RGB后显示")
plt.axis('off')
plt.show()

# 5. 转换为灰度图并显示
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(gray_img, cmap='gray')
plt.title("灰度图像")
plt.axis('off')
plt.show()

# 6. 添加噪声
# 椒盐噪声
sp_noise_img = random_noise(rgb_img, mode='s&p', amount=0.3)
# 高斯噪声
gus_noise_img = random_noise(rgb_img, mode='gaussian', mean=0, var=0.01)

# 显示原始图与噪声图对比
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(rgb_img)
plt.title("原始图像 (RGB)")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(sp_noise_img)
plt.title("添加椒盐噪声")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(gus_noise_img)
plt.title("添加高斯噪声")
plt.axis('off')

plt.tight_layout()
plt.show()

# 7. 应用滤波器
# 注意：random_noise输出的图像是[0,1]范围的浮点数，OpenCV滤波需要转换为[0,255]的uint8
sp_noise_uint8 = (sp_noise_img * 255).astype(np.uint8)
gus_noise_uint8 = (gus_noise_img * 255).astype(np.uint8)

# 均值滤波
mean_3_sp = cv2.blur(sp_noise_uint8, (3, 3))
mean_3_gus = cv2.blur(gus_noise_uint8, (3, 3))

# 中值滤波 (使用OpenCV内置函数)
mid_3_sp = cv2.medianBlur(sp_noise_uint8, 3)
mid_3_gus = cv2.medianBlur(gus_noise_uint8, 3)

# 8. 显示滤波结果对比 (椒盐噪声)
plt.figure(figsize=(12, 4))

plt.subplot(1, 4, 1)
plt.imshow(sp_noise_img)
plt.title("椒盐噪声原图")
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(mean_3_sp, cmap='gray')
plt.title("均值滤波 (3x3)")
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(mid_3_sp, cmap='gray')
plt.title("中值滤波 (3x3)")
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(gus_noise_img)
plt.title("高斯噪声对比")
plt.axis('off')

plt.tight_layout()
plt.show()

# 9. 自定义中值滤波函数（根据您提供的代码实现）
def median_filter_single_channel(src: np.ndarray, ksize: int) -> np.ndarray:
    """
    单通道图像的中值滤波
    
    参数:
        src: 输入的单通道图像 (灰度图)
        ksize: 滤波核大小，必须为奇数
        
    返回:
        中值滤波后的图像
    """
    h, w = src.shape
    pad = ksize // 2
    
    # 边界填充(使用reflect模式)
    padded = np.pad(src, pad, mode='reflect')
    
    # 执行中值滤波
    result = np.zeros((h, w), dtype=src.dtype)
    
    for i in range(h):
        for j in range(w):
            # 提取窗口
            window = padded[i:i + ksize, j:j + ksize]
            # 计算中值
            result[i, j] = np.median(window)
    
    return result

# 测试自定义中值滤波函数
if len(gray_img.shape) == 2:
    custom_mid_3 = median_filter_single_channel(gray_img, 3)
    
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(gray_img, cmap='gray')
    plt.title("原始灰度图像")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(custom_mid_3, cmap='gray')
    plt.title("自定义中值滤波 (3x3)")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

print("图像处理流程完成！")

