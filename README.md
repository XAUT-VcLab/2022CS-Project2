# ProDecouTalk：结合分阶段解耦和向量量化扩散的情感可控3D说话人脸生成方法
## 简介
项目位置：https://github.com/chen114514sheng/ProDecouTalk

本项目使用语音、文本、身份向量这3个条件，结合分阶段解耦与向量量化扩散的方法生成说话人脸视频。
## 方法概述
项目整体流程如下：

第一阶段
<img width="23350" height="13013" alt="图1" src="https://github.com/user-attachments/assets/e588e536-669e-4f81-92bf-33d61652bfcc" />
第二阶段
<img width="22675" height="8975" alt="图2" src="https://github.com/user-attachments/assets/5551c79e-ad6c-4533-b849-b8314538c8fd" />
生成阶段
<img width="19100" height="7367" alt="图3" src="https://github.com/user-attachments/assets/2ad6a6c0-4770-4ea3-8009-1a8d6e3956e0" />
## 数据集
3D人脸数据来自3DMEAD，语音数据来自MEAD，文本数据来自TA-MEAD。

数据集预处理：
```
DataProcess/mead0.py
```
数据集划分：
```
DataProcess/mead1.py
```
## 训练过程
训练第一阶段模型：
```
VQVAE2/Train.py
```
训练第二阶段模型：
```
Diffusion/Train.py
```
## 测试与生成
评估第一阶段模型的重建效果：
```
VQVAE2/Predict.py
```
评估第二阶段模型的生成效果，保存生成结果：
```
Diffusion/Predict.py
```
生成说话人脸视频：
```
Render.py #也可用于生成其他模型的说话人脸视频
```
本模型与其他模型比较：
```
Quality.py #生成heatmap
```
## 注意事项
模型参数、权重保存路径、文件路径通过以下文件设置：
```
config.yaml
```
以下文件中的路径同样需要设置：
```
DataProcess/mead0.py
DataProcess/mead1.py
Render.py
Quality.py
```
