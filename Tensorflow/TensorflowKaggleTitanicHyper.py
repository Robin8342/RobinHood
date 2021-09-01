import pandas as pd
import numpy as np
import os
from tensorflow import keras
from tensorflow.keras.layers import Dense
from copy import copy
#Dropout과 BatchNormalization을 추가한다.
from tensorflow.keras.layers import (Dense, Dropout, BatchNormalization)


file_path = "KaggleImage/titanic"
remove_list = ["Name", "Ticket"]

def import_Data(file_path):
    result = dict()
    for file in os.listdir(file_path):
        file_name = file[:-4]
        result[file_name] = pd.read_csv(file_path + "/" + file)

    return result

def make_Rawdata(dict_data):
    dict_key = list(dict_data.keys())
    test_Dataset = pd.merge(dict_data["gender_submission"], dict_data["test"], how='outer', on='PassengerId')
    Rawdata = pd.concat([dict_data["train"], test_Dataset])
    Rawdata.reset_index(drop=True, inplace=True)

    return Rawdata

def remove_columns(DF, remove_list):

    result = copy(Rawdata)
    result.set_index("PassengerId", inplace = True)

    for column in remove_list:
        del(result[column])

    return result


def missing_value(DF):
    del(DF["Cabin"])
    DF.dropna(inplace = True)


#One-Hot 벡터
def one_hot_Encoding(data, column):
    freq = data[column].value_counts()
    vocabulary = freq.sort_values(ascending = False).index
    for word in vocabulary:
        new_column = column + "_" + str(word)
        data[new_column] = 0

    for word in vocabulary:
        target_index = data[data[column] == word].index
        new_column = column + "_" + str(word)
        data.loc[target_index, new_column] = 1

    del(data[column])

def scale_adjust(X_test, X_train, C_number, key ="min_max"):
    if key == "min_max":
        min_key = np.min(X_train[:,C_number])
        max_key = np.max(X_train[:,C_number])

        X_train[:,C_number] = (X_train[:,C_number] - min_key)/(max_key - min_key)
        X_test[:,C_number] = (X_test[:,C_number] - min_key)/(max_key - min_key)

    elif key =="norm":
        mean_key = np.mean(X_train[:,C_number])
        std_key = np.std(X_train[:,C_number])

        X_train[:,C_number] = (X_train[:,C_number] - mean_key)/std_key
        X_test[:,C_number] = (X_test[:,C_number] - mean_key)/std_key

    return X_test, X_train

Rawdata_dict = import_Data(file_path)
Rawdata = make_Rawdata(Rawdata_dict)

#불필요한 Column 제거
DF_Hand = remove_columns(Rawdata, remove_list)

missing_value(DF_Hand)

one_hot_Encoding(DF_Hand, 'Pclass')
one_hot_Encoding(DF_Hand, 'Sex')
one_hot_Encoding(DF_Hand, 'Embarked')

y_test, y_train = DF_Hand["Survived"][:300].to_numpy(), DF_Hand["Survived"][300:].to_numpy()

del(DF_Hand["Survived"])
X_test, X_train = DF_Hand[:300].values, DF_Hand[300:].values

X_test, X_train = scale_adjust(X_test, X_train, 0, key="min_max")
X_test, X_train = scale_adjust(X_test, X_train, 1, key="min_max")
X_test, X_train = scale_adjust(X_test, X_train, 2, key="min_max")
X_test, X_train = scale_adjust(X_test, X_train, 3, key="min_max")

model = keras.Sequential()
model.add(BatchNormalization())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.10))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.10))
model.add(Dense(32, activation="relu"))
model.add(Dropout(0.10))
model.add(Dense(16, activation="relu"))
model.add(Dropout(0.50))
model.add(Dense(1, activation="sigmoid"))

opt = keras.optimizers.Adam(learning_rate = 0.005)
model.compile(optimizer = opt,
              loss = "binary_crossentropy",
              metrics=["binary_accuracy"])

model.fit(X_train, y_train, epochs = 200)

#모델학습 표현을 evaluate로 복잡하게 작성했던 코드를 한 줄로 대체가 가능하다.
#test_loss 는 손실값, test_acc 는(accuracy)를 의미한다.
test_loss, test_acc = model.evaluate(X_test, y_test, verbose = 2)

#에폭치를 100으로 했더니 80%의 효율로 올라갔다. 에폭치가 많아서 과적합이여서 성능이 느려진걸 수도 있다.
#Dropout과 Batchnormalization을 사용한 결과 82%로 2%가 더 상승했다.
#모델의 에폭치에 따라 전혀 다른 결과가 나올 수 있다.
#Dropout과 Batchnormalization이 Overfitting문제를 해결해 주므로 에폭치가 낮아서 성능이 느릴 수도 있다.
#물론 러닝 할때마다 효율이 높거나 낮을 수 있으나 Overfitting 문제를 해결하고 에폭치를 상승시킬 경우 효율이 조금 더 증가 한다.

#Dropout과 Batchnormalization을 모델에 추가 할 수 있따.
#Dropout은 Overfitting, model combination 문제를 해결하기 위해 등장한 개념이다.
#즉 신경망의 뉴런을 랜덤하게 부분적으로 생략시키는 방식이다.
#Batchnormalization은 배치 정규화로 활성화된 함수의 값이나 출력 값을 정규분포로 만들어서 초기화(가중치) 문제의 영향을 덜 받게 해준다.
#학습율(Learning Rate)을 높게 설정할 수 있어 학습 속도가 개선된다.
#Overfitting 위험을 줄일 수 있으며 가중치 소실(Gradient Vanishing) 문제를 해결해 준다.
print("Accuracy:", np.round(test_acc, 5))



