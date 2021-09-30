import urllib.request

import matplotlib.pyplot as plt
import numpy as np
import os

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator


#os.path.dirname 경로 중 디렉토리명만 얻기
#os.path.join path들을 묶어 하나의 경로로 만들기
PATH = os.path.join("./cats_and_dogs_filtered")

#os.listdir path 하위에 있는 파일, 디렉토리 리스트를 보여준다.
os.listdir(PATH)

#train과 validattion에 대한 파일 확장자를 나누고
train_path = os.path.join(PATH, 'train')
validation_path = os.path.join(PATH, 'validation')

#image의 rescale = 픽셀값을 조정한다.
original_datagen = ImageDataGenerator(rescale=1./255)

"""
rescale = 픽셀값
rotation_range = 이미지 회전
width_shift_range = 가로 방향 이동
height_shift_range = 세로 방향 이동
shear_range = 이미지 굴절
zoom_range = 이미지 확대
horizontal_flip = 횡 방향으로 이미지 반전
fill_mode = 이미지를 이동이나 굴절시 빈 픽셀 값을 채우는 방식
"""
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

for x, y in original_generator:
    pic = x[:5]
    break

conv2d = Conv2D(64, (3, 3), input_shape=(150, 150, 3))
conv2d_activation = Conv2D(64, (3, 3), activation='relu', input_shape=(150, 150, 3))

fig, axes = plt.subplots(8, 8)
fig.set_size_inches(16, 16)
for i in range(64):
    axes[i//8, i%8].imshow(conv2d(pic)[0,:,:,i], cmap='gray')
    axes[i//8, i%8].axis('off')

fig, axes = plt.subplots(8, 8)
fig.set_size_inches(16, 16)
for i in range(64):
    axes[i//8, i%8].imshow(conv2d_activation(pic)[0,:,:,i], cmap='gray')
    axes[i//8, i%8].axis('off')

fig, axes = plt.subplots(8, 8)
fig.set_size_inches(16, 16)
for i in range(64):
    axes[i//8, i%8].imshow(MaxPooling2D(2, 2)(conv2d(pic))[0, :, :, i], cmap='gray')
    axes[i//8, i%8].axis('off')


#build model
model = Sequential([
    Conv2D(16, (3, 3), padding='same', activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(32, (3, 3), padding='same', activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), padding='same', activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid'),
])

#지워도댐
model.summary()

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