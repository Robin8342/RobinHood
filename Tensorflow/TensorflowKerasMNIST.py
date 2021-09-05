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

#print(train_images.shape)
#print(train_labels.shape)
#print(test_images.shape)
#print(test_labels.shape)

#train_labels의 개수를 알 수 있다.
#print(pd.Series(train_labels).value_counts())

#개수를 show로 볼 수 있다.(그래프로 편하게 봄)
#평탄한 그래프의 모습
#plt.hist(train_labels)
#plt.show()


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

np.random.seed(1234)
index_list = np.arange(0, len(train_labels))
valid_index = np.random.choice(index_list, size = 5000, replace = False)

valid_images = train_images[valid_index]
valid_labels = train_labels[valid_index]

train_index = set(index_list) - set(valid_index)
train_images = train_images[list(train_index)]
train_labels = train_labels[list(train_index)]

#표준편차 두개의 값이 유사하게 나온다면 검증 셋의 대표성을 갖고 있다고 할 수 있다.
print(np.std(train_labels))
print(np.std(valid_labels))


#min-max scaling을 이용해 표준화.
min_key = np.min(train_images)
max_key = np.max(train_images)

train_images = (train_images - min_key)/(max_key - min_key)
valid_images = (valid_images - min_key)/(max_key - min_key)
test_images = (test_images - min_key)/(max_key - min_key)


#FlattenLayer -> 2차원 배열을 1차원 배열로 만들어주는 전처리용 Layer
#Row (1,28,28) 데이터셋을 reshape(-1,28*28) 형으로 변환시킨다.
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape = [28,28], name="Flatten"))
model.add(Dense(300, activation="relu", name ="Hidden1"))
model.add(Dense(200, activation="relu", name ="Hidden2"))
model.add(Dense(100, activation="relu", name ="Hidden3"))
model.add(Dense(10, activation="softmax", name ="Output"))

"""
#reshape 확인  (1,784) 즉 Layer은 Row에 적용되서 (60000, 28,28)에서 (60000,784)로 바뀐다.
X = train_images[0]
print(X.reshape(-1,28*28).shape)
"""

#모델의 요약 정보를 얻을 수 있다. 784* 300 = 235200 +300 = 235500
#파라미터의 수가 한곳에 많은 경우 과대적합(Overfitting) 위험이 올라갈 수 있으며 특히 데이터 양이 많을 경우 과대 적합의 위험이 올라간다.
#model.summary()


#show_images(train_images, train_labels, 4, 5)

#print(train_images[0])

#optimizer을 어떤 방법으로 선택하느냐에 따라 최적해를 찾아가는 속도가 다르다.
opt = keras.optimizers.Adam(learning_rate=0.005)
model.compile(optimizer = opt,
              loss = "sparse_categorical_crossentropy",
              metrics = ["accuracy"])

"""#콜백이 없는 모델 학습
history = model.fit(train_images, train_labels,
                    epochs = 30,
                    batch_size = 5000,
                    validation_data=(valid_images, valid_labels))
"""

#callbacks(콜벡) 조기종료 : 손실 값이 최소가 되는 순간 학습을 멈추는 방법
#monitor : 관찰할 값, mindelta : 개선 기준 최소 변화량, patience : 정지까지 기다리는 epochs (최솟값이 나왔다 할지라도, 10번 더 학습을 실시함)
#restore_best_weights : 최선 값이 발생한 때로 모델 가중치 복원 여부, False일 경우 학습 마지막 단계의 모델 가중치 사용
#러닝 해보면 자동으로 epochs = 22에서 멈추며 최선의 값을 얻게 된다.
#매우 중요!
early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=10, restore_best_weights=True)
history = model.fit(train_images, train_labels,
                    epochs=100,
                    batch_size = 5000,
                    validation_data = (valid_images, valid_labels),
                    callbacks=[early_stop])

"""
#loss율과 accuracy의 데이터를 시각화해서 볼 수 있다.
history_DF = pd.DataFrame(history.history)
#print(history_DF)

#그래프의 크기와 선의 굵기 설정
history_DF.plot(figsize=(12,8), linewidth=3)

plt.grid(True)

plt.legend(loc = "upper right", fontsize = 15)
plt.title("Learning Curve", fontsize=30, pad = 30)
plt.xlabel("Epoch", fontsize=20, loc="center", labelpad = 20)
plt.ylabel("variable", fontsize = 20, rotation = 0, loc="center", labelpad = 40)

#테두리 제거
ax = plt.gca()
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)

#위의 그래프를 이용해서 적절한 Epoch 값을 구할 수 있다.
#그래픽에서 보면 val_loss(손실값)가 감소하다 소폭 상승하는 곳이 있다. 그곳이 과대적합이 되는 곳이니 해당 그래프의 이전 에포치를 선택하면 된다.
#감소하다 소폭 상승하는 이유는 지나치게 최적화되어 일어난 과대적합이다.
plt.show()
"""
model.save("./Model/data/MNIST_202108.h5")

