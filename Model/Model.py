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