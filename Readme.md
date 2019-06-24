# 计算机视觉大作业报告

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

对应代码如下所示

![1561365352888](C:\Users\Jacob\AppData\Roaming\Typora\typora-user-images\1561365352888.png)

## 模型

Predict/Predict.py

网络结构为InceptionV3，后面添加了平均池化层、DropOut层、展开Flatten层和一层全连接网络，最后的输出为120的神经元，利用Softmax函数进行分类。新加的网络结构示意图如下

![1561376195740](C:\Users\Jacob\AppData\Roaming\Typora\typora-user-images\1561376195740.png)

先固定住InceptionV3网络的权重，训练后面新加的权重。此时采用的迭代算法为rmsprop，迭代5次即可。

之后再把所有层设置为可训练的，对整个网络进行训练。此时采用的训练算法为sgd，学习率也取得比较小，为1e-4。

具体的代码如下所示

```python
import os
import keras
import numpy as np
from keras import Input
from keras import backend as K
from keras.applications import Xception,InceptionV3
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Dense, Dropout, Lambda,AveragePooling2D,Flatten
from keras.models import Model, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.utils import plot_model

img_rows,img_cols,img_channel=299,299,3
train_datagen = ImageDataGenerator(
        rotation_range=30, 
        rescale=1./255,
        shear_range=0.3,
        zoom_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2, 
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)
early_stopping = EarlyStopping(monitor='val_loss', patience=3)
auto_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=0, mode='auto', epsilon=0.001, cooldown=0, min_lr=0)
save_model = ModelCheckpoint('InceptionV3{epoch:02d}-{val_acc:.2f}.h5', period=2)

# Create the base pre-trained model
input_tensor = Input(shape=(img_rows, img_cols, img_channel))
base_model = InceptionV3(weights='imagenet', include_top=False,input_shape=(img_rows, img_cols, img_channel))
x=AveragePooling2D(pool_size=(4,4))(base_model.output)
x=Dropout(0.5)(x)
x=Flatten()(x)
x=Dense(120, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=x)

# 锁住所有 InceptionV3 的卷积层
for layer in base_model.layers:
    layer.trainable = False

batch_size = 8
epoch=5
train_generator = train_datagen.flow_from_directory(
    '../data/train',
    target_size=(img_rows, img_cols),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    '../data/val',
    target_size=(img_rows, img_cols),
    batch_size=batch_size,
    class_mode='categorical')

# 用RMSPROP方法，迭代五次，粗训练
model.compile(loss="categorical_crossentropy", optimizer='rmsprop', metrics=['accuracy'])

model.fit_generator(train_generator,
                steps_per_epoch=train_generator.samples/batch_size+1,
                epochs=epoch,
                validation_data=validation_generator,
                validation_steps=validation_generator.samples/batch_size+1,
                callbacks=[early_stopping, auto_lr, save_model]
                )

# 把所有层都设为可训练的			
for layer in model.layers:
    layer.trainable = True

model.summary()          
batch_size=8
epoch=30
train_generator = train_datagen.flow_from_directory(
	'../data/train',
	target_size=(img_rows, img_cols),
	batch_size=batch_size,
	class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
	'../data/val',
	target_size=(img_rows, img_cols),
	batch_size=batch_size,
	class_mode='categorical')
# 用SGD，选择一个小的lr，细训练
model.compile(loss="categorical_crossentropy", optimizer=SGD(lr=1e-4, momentum=0.9), metrics=['accuracy'])
save_model = ModelCheckpoint('InceptionV3-{epoch:02d}-{val_acc:.2f}.h5', monitor='val_acc')
model.fit_generator(train_generator,
                steps_per_epoch=train_generator.samples/batch_size+1,
                epochs=epoch,
                validation_data=validation_generator,
                validation_steps=validation_generator.samples/batch_size+1,
                callbacks=[auto_lr, save_model])
model.save('inceptionV3.h5')
```

## 生成csv文件

Model/Model.py

代码编写很简单，代码如下所示

```Python
import os
import operator
import shutil
import cv2
import h5py
import numpy as np
import pandas as pd
from keras.models import load_model 
from keras.preprocessing.image import ImageDataGenerator
model_path="../models/InceptionV3-30-0.88.h5"
test_data_dir="../data/test"
val_data_dir="../data/val"
model=load_model(model_path)
test_datagen = ImageDataGenerator(rescale=1./255)
batch_size=64

val_generator = test_datagen.flow_from_directory(
    val_data_dir,
    target_size=(299, 299),
    batch_size=batch_size,
    shuffle=False,
    class_mode='categorical'
)
label_idxs = sorted(val_generator.class_indices.items(), key=operator.itemgetter(1))
test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(299, 299),
        batch_size=batch_size,
        shuffle=False,
        class_mode='categorical')

y= model.predict_generator(test_generator, test_generator.samples/batch_size)
y_max_idx = np.argmax(y, 1)
res_path = '1801213637.csv'

with open(res_path,'w') as f:
    f.write("Id,Category\n")
    for i, idx in enumerate(y_max_idx):
        f.write(test_generator.filenames[i][5:]+ ',' +str(label_idxs[idx][0]) + '\n')
```

## 结果

最后的分类正确率在88%左右。

## Reference

代码撰写参考了 https://github.com/holyhao/Baidu-Dogs，在其基础上做了一系列修改。 
