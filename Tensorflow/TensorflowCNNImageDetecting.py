import urllib.request
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

