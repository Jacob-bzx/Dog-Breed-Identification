# 计算机视觉大作业

## 比赛要求

比赛地址：https://www.kaggle.com/c/dog-classification

 用120 个种类、共计15,780 张图片的狗的数据集，训练一个网络，对测试集中的 4800 张狗的图片进行分类。

## 硬件环境

- GeForce GTX 2060 6G
- Inter i7-8700
- 内存：32G

## 主要软件环境

* tensorflow-gpu 1.13.1
* keras 2.2.4

## 数据预处理

Preprocessing/SplitTrainValid.py

将官网上的数据集下载、解压。

将训练集的数据中提取20%作为验证集，并将其从训练集中删除。

## 模型

Predict/Predict.py

网络结构为InceptionV3，后面添加了平均池化层、DropOut层、展开Flatten层和一层全连接网络，最后的输出为120的神经元，利用Softmax函数进行分类。

先固定住InceptionV3网络的权重，训练后面新加的权重。此时采用的迭代算法为rmsprop，迭代5次即可。

之后再把所有层设置为可训练的，对整个网络进行训练。此时采用的训练算法为sgd，学习率也取得比较小，为1e-4。

## 生成csv文件

Model/Model.py

代码逻辑很简单。

## 结果

最后的分类正确率在88%左右。

## Reference

代码参考了 [holyhao](https://github.com/holyhao/Baidu-Dogs) ，在其基础上做了一 点修改。 

