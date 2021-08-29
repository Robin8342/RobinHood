import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Dense

def f1(x1, x2, x3, x4):
    return 0.3 *x1 + 0.2*x2 - 0.4*x3 + 0.1*x4 +2

def f2(x1, x2, x3, x4):
    return 0.5 *x1 + 0.1*x2 - 0.3*x3 -2

def make_dataset(stert_N, end_N):
    x1 = np.arange(stert_N, end_N)
    x2 = x1 + 1
    x3 = x1 + 2
    x4 = x1 + 3

    y1 = f1(x1, x2, x3, x4)
    y2 = f2(x1, x2, x3, x4)

    append_for_shuffle = np.c_[x1,x2,x3,x4,y1,y2]
    np.random.shuffle(append_for_shuffle)

    X = append_for_shuffle[:,[0,1,2,3]]
    Y = append_for_shuffle[:,[4,5]]

    return X, Y


X, Y = make_dataset(0, 1000)
X_train, X_test = X[:800], X[800:]  #train과 test를 8:2로 나눔(7:3이 평균적)
Y_train, Y_test = Y[:800], Y[800:]


#model make
model = keras.Sequential()
model.add(Dense(128, activation = "relu"))
model.add(Dense(64, activation = "relu"))
model.add(Dense(32, activation = "relu"))
model.add(Dense(16, activation = "relu"))
model.add(Dense(2, activation = "linear"))

opt = keras.optimizers.Adam(learning_rate = 0.005)
model.compile(optimizer = opt, loss = "mse")

#표준화
min_key = np.min(X_train)
max_key = np.max(X_train)

X_std_train = (X_train - min_key)/(max_key - min_key)
Y_std_train = (Y_train - min_key)/(max_key - min_key)
X_std_test = (X_test - min_key)/(max_key - min_key)

model.fit(X_std_train, Y_std_train, epochs = 100)


def MAE(x, y):
    return np.mean(np.abs(x-y))

pred = model.predict(X_std_test) * (max_key - min_key) + min_key
print("Accuracy:", MAE(pred,Y_test))
