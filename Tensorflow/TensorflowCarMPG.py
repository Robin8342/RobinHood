import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
#train_test_split 함수를 통해 단 1줄로 깔끔하게 train / test을 나눌 수 있다.
#단 slkearn의 경우 모든 데이터에 대한 정보가 있는 상태가 아니여서 데이터 파악부터 해야 된다.
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np

#One-Hot Vector
def make_One_Hot(data_DF, column):
    target_column = data_DF.pop(column)
    for i in sorted(target_column.unique()):
        new_column = column + "_" + str(i)
        #1.0을 곱하여 int 로 바꿔주었다. ( Python 은 1.0을 곱해주어도 int형으로 변하게 된다.)
        data_DF[new_column] = (target_column ==i) * 1.0


#tf.keras.utils.get_file()을 이용해서 데이터셋을 쉽게 가져올 수 있다.

dataset_path = keras.utils.get_file("auto-mpg.data","https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")

#dataset이
column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                'Acceleration','Model Year','Origin']

#read_csv의 파일을 불러와서 \t만큼 띄어서 확인할 수 있다.
Rawdata = pd.read_csv(dataset_path, names=column_names, na_values = "?",
                      comment='\t', sep=" ", skipinitialspace=True)

print(Rawdata)

#데이터가 누락되어있는지 확인
Rawdata.isna().sum()

Rawdata.dropna(inplace = True)

print("결측값 비율", Rawdata.isna().sum(1).sum()/len(Rawdata))

label = Rawdata.pop("MPG").to_numpy()
dataset = Rawdata.values
X_train, X_test, y_train, y_test = train_test_split(dataset, label, test_size = 0.3)

mean_point = X_train[:,:6].mean(axis=0)
std_point = X_train[:,:6].std(axis=0)

X_train[:,:6] = ((X_train[:,:6] - mean_point)/std_point)
X_test[:,:6] = ((X_test[:,:6] - mean_point)/std_point)

model = keras.Sequential([
    keras.layers.Dense(60, activation = "relu"),
    keras.layers.Dense(1, activation = "linear")
])

opt = keras.optimizers.Adam(learning_rate = 0.005)
model.compile(optimizer=opt, loss="mse", metrics = ["accuracy"])

early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
model.fit(X_train, y_train, epochs=300, validation_split=0.2, callbacks=[early_stop])

print("===============")
print(model.evaluate(X_test,y_test))