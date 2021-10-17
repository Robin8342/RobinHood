#모델 학습 데이터를 미리 학습한다고 해도. EPOCH에 넣어서 이미지를 테스팅 해야되는건 똑같다. 물론 도커에 올려서 비교할 수 있는 방법도 있다.


import urllib.request
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras.models import load_model

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

model = load_model('./TestFolder/TensorflowCNNCreateData.h5')
PATH = os.path.join("./cats_and_dogs_filtered")

#os.listdir path 하위에 있는 파일, 디렉토리 리스트를 보여준다.
os.listdir(PATH)

#train과 validattion에 대한 파일 확장자를 나누고
train_path = os.path.join(PATH, 'train')
validation_path = os.path.join(PATH, 'validation')

#image의 rescale = 픽셀값을 조정한다.
original_datagen = ImageDataGenerator(rescale=1./255)

training_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest')

original_generator = original_datagen.flow_from_directory(train_path,
                                                          batch_size=128,
                                                          target_size=(150, 150),
                                                          class_mode='binary'
                                                         )

training_generator = training_datagen.flow_from_directory(train_path,
                                                          batch_size=128,
                                                          shuffle=True,
                                                          target_size=(150, 150),
                                                          class_mode='binary'
                                                         )

validation_generator = training_datagen.flow_from_directory(validation_path,
                                                            batch_size=128,
                                                            shuffle=True,
                                                            target_size=(150, 150),
                                                            class_mode='binary'
                                                           )

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

epochs = 25

history = model.fit(training_generator,
                    validation_data= validation_generator,
                    epochs=epochs)

#학습결과 확인
plt.figure(figsize=(9, 6))
plt.plot(np.arange(1, epochs+1), history.history['loss'])
plt.plot(np.arange(1, epochs+1), history.history['val_loss'])
plt.title('Loss / Val Loss', fontsize=20)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['loss', 'val_loss'], fontsize=15)
plt.show()


