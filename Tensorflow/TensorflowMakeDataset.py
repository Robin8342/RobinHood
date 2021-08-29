import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.layers import Dense

def f(x):
    return x + 10

np.random.seed(1234)
X_train = np.random.randint(0, 100,(100,1))
X_test = np.random.randint(100,200,(20,1))

y_train = f(X_train)
y_test = f(X_test)


#model activation을 relu 로 했을 때보다 단순한 계산법에서 linear을 사용하니 신뢰도가 더욱 올라갔다.
#이는 하이퍼 파라미터를 grred search 나 random search , bayesian optimization등 처럼 더 좋은 해들의 조합해서 처리 하는 편이 좋다.
#같은 알고리즘이더라도 데이터와 활성화 함수, 손실 함수를 무엇을 쓰느냐에 따라서 결과가 무수히 많이 달라질 수 있다.
model = keras.Sequential()
model.add(Dense(16, activation='linear'))
model.add(Dense(1, activation='linear'))

opt = keras.optimizers.Adam(learning_rate = 0.01)
model.compile(optimizer = opt, loss ='mse')

mean_key = np.mean(X_train)
std_key = np.std(X_train)

X_train_std = (X_train - mean_key)/std_key
y_train_std = (y_train - mean_key)/std_key
X_test_std = (X_test - mean_key)/std_key


model.fit(X_train_std,y_train_std, epochs = 100)

model.predict(X_test.reshape((X_test.shape[0])))

pred = model.predict(X_test_std.reshape((X_test_std.shape[0])))

pred_restore = pred * std_key + mean_key
predict_DF = pd.DataFrame({"predict":pred_restore.reshape(pred_restore.shape[0]), "label":y_test.reshape(y_test.shape[0])})
predict_DF["gap"] = predict_DF["predict"] - predict_DF["label"]



print("Accuracy:", np.sqrt(np.mean((pred_restore - y_test)**2)))