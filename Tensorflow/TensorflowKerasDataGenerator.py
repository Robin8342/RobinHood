import tensorflow as tf
import numpy as np
from tensorflow.python.keras.utils.data_utils import Sequence
import cv2

class DataGenerator(Sequence):
    def __init__(self, path, list_IDs, labels,
                 batch_size, img_size, img_channel, num_classes):

        # 데이터셋 경로
        self.path = path
        # 데이터 이미지 개별 주소 [ DataFrame 형식 (image 주소, image 클래스) ]
        self.list_IDs = list_IDs
        # 데이터 라벨 리스트 [ DataFrame 형식 (image 주소, image 클래스) ]
        self.labels = labels
        # 학습 Batch 사이즈
        self.batch_size = batch_size
        # 이미지 리사이징 사이즈
        self.img_size = img_size
        # 이미지 채널 [RGB or Gray]
        self.img_channel = img_channel
        # 데이터 라벨의 클래스 수
        self.num_classes = num_classes
        # 전체 데이터 수
        self.indexes = np.arange(len(self.list_IDs))

    def __len__(self):
        len_ = int(len(self.list_IDs) / self.batch_size)
        if len_ * self.batch_size < len(self.list_IDs):
            len_ += 1
        return len_

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def __data_generation(self, list_IDs_temp):
        X = np.zeros((self.batch_size, self.img_size, self.img_size, self.img_channel))
        y = np.zeros((self.batch_size, self.num_classes), dtype=int)
        for i, ID in enumerate(list_IDs_temp):
            img = cv2.imread(self.path + ID)
            img = cv2.resize(img, (self.img_size, self.img_size))
            X[i,] = img / 255
            y[i,] = tf.keras.utils.to_categorical(self.labels[i], num_classes=self.num_classes)
        return X, y

"""
import pandas as pd

# 이미지 주소 및 클래스 라벨 파일 불러오기
train_labels = pd.read_csv('train.csv')

# 라벨 정보 전처리
# 전체 클래스 수
clss_num = len(train_labels['labels'].unique())
# 클래스 -> 숫자로 변환 (카테고리 형식의 클래스를 원 핫 인코딩)
labels_dict = dict(zip(train_labels['labels'].unique(), range(clss_num)))
train_labels = train_labels.replace({"labels": labels_dict})

tartget_size = 150
img_ch = 3
num_class = 12
batch_size = 32

train_generator = DataGenerator('train_images/', train_labels['image'],
                                train_labels['labels'],
                                batch_size, tartget_size,
                                img_ch, num_class)
 
# 학습
# history = model.fit_generator(train_generator, epochs=1)


"""