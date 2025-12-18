# 实验 2 ：图像增强



## 实验目的
学会 Opencv 的基本使用方法，利用 Opencv 等计算机视觉库对图像进行平滑、滤波等操作，实现图像增强。

## 实验内容
2.1  导入图像滤波相关的依赖包

<img width="1108" height="282" alt="image" src="https://github.com/user-attachments/assets/128515f4-9cbc-42c1-924e-a56d132f4bd4" />


2.2  读取原始图像并进行色彩空间转换

<img width="1111" height="352" alt="image" src="https://github.com/user-attachments/assets/501bc869-ad62-4892-88c1-9e0ba8a9db4d" />
<img width="1110" height="316" alt="image" src="https://github.com/user-attachments/assets/6cf567b3-3e78-49c2-8622-0713a7826259" />
<img width="1110" height="311" alt="image" src="https://github.com/user-attachments/assets/ba62f3d1-8f49-449b-a41e-ae33d6533ee1" />




2.3  添加噪声

<img width="1110" height="449" alt="image" src="https://github.com/user-attachments/assets/1fe78d57-43d9-48b5-a726-775f6a084d2e" />

<img width="1112" height="433" alt="image" src="https://github.com/user-attachments/assets/4cc6464f-71e4-4039-abca-5eceabe8e352" />

<img width="1110" height="512" alt="image" src="https://github.com/user-attachments/assets/bdc6449e-7caf-4186-90aa-573927b4be14" />





2.4  图像滤波
<img width="1108" height="552" alt="image" src="https://github.com/user-attachments/assets/c01b9ea8-c8e9-4078-a9a2-9c226e1dbd1f" />
<img width="1111" height="671" alt="image" src="https://github.com/user-attachments/assets/8ee4583a-e990-480b-a5f8-eb89a315deb7" />

## 实验总结

这次实验的核心是认识两种常见图像噪声，对比不同滤波方法的去噪效果，用专业指标评估效果好坏，并解决实验中遇到的代码和操作问题。

首先是认识两种噪声。高斯噪声看起来像图像上蒙了一层雪花状的灰色颗粒，分布比较均匀；椒盐噪声则是图像中随机出现的黑色和白色小点，就像信号传输时出了错误导致的。清楚它们的样子，才能选对合适的滤波方法。

然后是测试不同滤波器的效果。我给图像添加噪声后，分别用了五种滤波方法处理：均值滤波能减轻高斯噪声，但会让图像变模糊，细节看不清楚；中值滤波对付椒盐噪声特别管用，能把黑白小点去掉，还能保住图像的边缘；高斯滤波比均值滤波的平滑效果更自然，也能减轻高斯噪声，但还是会有点模糊；双边滤波去噪的同时，能很好地保留图像的边缘和细节，看起来效果最好，但处理起来稍慢；非局部均值滤波对彩色图像的去噪效果不错，能留住纹理细节，就是花费的处理时间较长。

最后是效果的定量评估。我用了两个指标来衡量：信噪比（PSNR）和结构相似性（SSIM）。结果发现，所有滤波方法都能让这两个指标变好：中值滤波让椒盐噪声图像的 PSNR 提升得最明显；双边滤波的 SSIM 值最高，说明它保留图像细节的能力最强；非局部均值滤波处理彩色图像时，能更好地还原原本的颜色。

