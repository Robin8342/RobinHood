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

#호출한 파일의 Name.head를 입력시 데이터의 생김새를 볼 수 있으며 제일 처음 "," 뒤에는 인물의 이름이 나온다.
#print(Rawdata.Name.head(20))


#Name 카테고리와 Class 카테고리를 가져온다.
Class1 = Rawdata["Name"].str.partition(",")[2]
Rawdata["Class"] = Class1.str.partition(".")[0]

#Rawdata내의 Class 정보를 확인 한다. Mrs과 Miss 그리고 보면 Ms, Mile, Lady처럼 Miss에 속하는 정보가
#중복으로 되어 있는걸 볼 수 있다. 이를 하나로 묶으며 자료를 정리한다.
#print(Rawdata.Class.value_counts())

#Class_a에 들어간 데이터들을 보면 [' Mr' ' Mrs' ' Miss' ... ' Mr' ' Mr' ' Master']
#되어있다. 즉 앞에 띄어쓰기까지 해야 확실하게 디텍팅하며 합치거나 버리게 된다.
Class_a = Rawdata["Class"].to_numpy()

"""
print("##############")
print(Class_a)
"""

Class_b = np.where(Class_a == ' Mr', 0,
            np.where(np.isin(Class_a,[' Miss',' Mlle',' Ms',' Lady']),1,
                np.where(np.isin(Class_a, [' Mrs',' the Countess',' Dona', ' Mme']),2,9)))

Rawdata["Class"] = Class_b

"""
print("###########")
print(Class_b)
print("################")
#데이터 처리 된 이후에 확인
print(Rawdata["Class"].value_counts())
"""

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
model.add(Dense(3, activation="softmax"))

opt = keras.optimizers.Adam(learning_rate = 0.005)


#sparse_categorical_crossentropy 로 로스율을 측정했다. 근데 왜 로스율이 0.9363인데 accuracy가 82%지...??
#loss: 0.9363 - accuracy: 0.8233
model.compile(optimizer = opt,
              loss = "sparse_categorical_crossentropy",
              metrics=["accuracy"])

model.fit(X_train, y_train, epochs = 200)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose = 2)

print("Accuracy:", np.round(test_acc, 5))



