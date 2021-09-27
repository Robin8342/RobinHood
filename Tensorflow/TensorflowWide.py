#Deep Models : 정보의 일반화 -> 참새는 날 수 있다. -> 비둘기는 날 수 있다 -> 날개를 가진 동물은 날 수 있다.
#Wide Models : 정보의 암기
#Wide & Deep Learning : 추천 시스템, 검색 및 순위 문제 같은 많은 양의 범주형 특징이 있는 데이터를 사용할 때 사용된다.
#복잡한 패턴과 간단한 규칙 모두 학습할 수 있다.
#단 wide Deep Learning은 keras 함수형 api을 이용해서 만들어야 된다.
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import (Input, Dense, Concatenate)



Rawdict = fetch_california_housing()

#DataFrame으로 Rawdict의 데이터를 가져온다.
Cal_DF = pd.DataFrame(Rawdict.data, columns = Rawdict.feature_names)

"""
#데이터셋을 가져오면 꼭 데이터 확인과 데이터 타입을 확인해야된다. 그리고 결측값을 확인해야된다.
#만약 결측값이 생긴다면 해당 부분을 삭제해야된다.
print(Cal_DF)
print(Cal_DF.dtypes)
print(Cal_DF.isnull().sum()
"""

#train_test_split(array, test_size, shuffle) = sklearn으로 불러온 Rawdict 데이터를 이용해 (dataset, label) 두 개의 데이터를 동시에 넣고 나눈다.
X_train_all, X_test, y_train_all, y_test = train_test_split(Rawdict.data, Rawdict.target, test_size = 0.3)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_all, y_train_all, test_size = 0.2)

"""
#데이터 모양
print("Train shape", X_train.shape)
print("Validation", X_valid.shape)
print("Test set", X_test.shape)
"""

scaler = StandardScaler()
#StandardScaler() : 표준 정규분포, MinMaxScaler(): 최소,최대 스케일 변환
#MaxAbsScaler() : 최대 절댓값 1로 변환 (이상치 영향이 큼), RobustScaler() : StandardScaler보다 표준화 후 동일한 값을 넓게 분포
X_train = scaler.fit_transform(X_train)
#fit_transform : 데이터셋 표준 정규분포화. 단 dataset의 평균과 표준편차를 기준으로 저장하게 됨.

X_valid = scaler.transform(X_valid)
#transform() : 데이터셋을 표준 정규분포화. 평균과 표준편차는 fit 된 dataset을 따른다.
X_test = scaler.transform(X_test)

#Model 생성
inputData = Input(shape=X_train.shape[1:])
hidden1 = Dense(30, activation="relu")(inputData)
hidden2 = Dense(30, activation="relu")(hidden1)
concat = Concatenate()([inputData, hidden2])
output = Dense(1)(concat)
model = keras.Model([inputData], outputs=[output])

#Concatenate()([array1],[array2]) : axis을 어떻게 잡느냐에따라 출력되는 array의 모양을 다르게 만들 수 있다.

#모델 컴파일
model.compile(optimizer=Adam(learning_rate=0.005),
              loss = "msle",
              metrics=["accuracy"])

#학습
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)

history = model.fit(X_train, y_train,
                    epochs=100,
                    batch_size=32,
                    validation_data =(X_valid, y_valid),
                    callbacks=[early_stop])

print(model.evaluate(X_test, y_test))
