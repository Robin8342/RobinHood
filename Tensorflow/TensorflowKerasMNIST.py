#생각해보니 숫자 입력받는 어플 만들면 MNIST 쓰면 되네....
#예를들어 명함 사진 ? -> 근데 이건 OPENCV로 할 수 있음)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (Dense, BatchNormalization, Dropout)
from tensorflow.keras.datasets.mnist import load_data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = load_data()

print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_labels.shape)

def show_images(dataset, label, nrow, ncol):
    flg, axes = plt.subplots(nrows=nrow, ncols=ncol, figsize=(2*ncol,2*nrow))
    ax = axes.ravel()

    xlabels = label[0:nrow*ncol]

    for i in range(nrow*ncol):
        image = dataset[i]
        ax[i].imshow(image, cmap='gray')
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        ax[i].set_xlabel(xlabels[i])

    #tight_layout 으로 빈 칸 없이 이미지를 다 채운다.
    plt.tight_layout()
    plt.show()

show_images(train_images, train_labels, 4, 5)

print(train_images[0])