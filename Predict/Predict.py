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
