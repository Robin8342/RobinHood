#patten : 1/2x**2 - 3y + 5
#제곱승이라는 식때문에 간단한 선형 함수가 아니라 데이터 신뢰도가 엄청 낮아졌다.
#이를 위해 train Dataset 과 test Dataset 을 분리해야된다.
#Train dataset 과 test Dataset을 분리하기 전에는 정확도가 안좋았지만
#분리 이후 정확도는 훨씬 좋아졌다.

import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Dense

#dataset
np.random.seed(1234)

def f2(x1, x2):
    return 0.5*x1**2 - 3*x2 + 5

X1 = np.random.randint(0, 100, (1000))
X2 = np.random.randint(0, 100, (1000))
X = np.c_[X1, X2]
Y = f2(X1, X2)

#데이터셋을 중복되지 않게 나눈다.
Xy = np.c_[X,Y]
Xy = np.unique(Xy, axis = 0)
np.random.shuffle(Xy)
test_len = int(np.ceil(len(Xy)*0.3))
X = Xy[:, [0,1]]
Y = Xy[:, 2]

#test 셋과 train 셋
X_test = X[:test_len]
Y_test = Y[:test_len]

X_train = X[test_len:]
Y_train = Y[test_len:]



"""
X0_1 = np.random.randint(0, 100, (1000))
X0_2 = np.random.randint(0, 100, (1000))
X_train = np.c_[X0_1, X0_2]
Y_train = f2(X0_1,X0_2)

X1_1 = np.random.randint(100, 200, (300))
X1_2 = np.random.randint(100, 200, (300))

X_test = np.c_[X1_1, X1_2]
Y_test = f2(X1_1, X1_2)
"""

#make model

model = keras.Sequential()
model.add(Dense(16, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(16, activation = 'relu'))
model.add(Dense(1, activation= 'linear'))

#compile
opt = keras.optimizers.Adam(learning_rate = 0.005)
model.compile(optimizer=opt, loss='mse')

#standardization
mean_key = np.mean(X_train)
std_key = np.std(X_train)

X_train_std = (X_train - mean_key)/std_key
Y_train_std = (Y_train - mean_key)/std_key

X_test_std = (X_test - mean_key)/std_key


#model epochs

model.fit(X_train_std, Y_train_std, epochs = 100)

pred = (model.predict(X_test_std) * std_key) + mean_key
pred = pred.reshape(pred.shape[0])
print("Accuracy:", np.sqrt(np.sum((Y_test - pred)**2))/len(Y_test))

