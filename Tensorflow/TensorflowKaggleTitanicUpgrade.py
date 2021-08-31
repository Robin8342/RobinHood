import pandas as pd
import numpy as np
import os
from tensorflow import keras
from tensorflow.keras.layers import Dense
from copy import copy

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

Rawdata_dict = import_Data(file_path)
Rawdata = make_Rawdata(Rawdata_dict)

#불필요한 Column 제거
DF_Hand = remove_columns(Rawdata, remove_list)

missing_value(DF_Hand)

#범주형 변수 -> 물체를 숫자로 치환한다 할지라도 문자형 데이터는 단순하게 숫자로 치환해주는 걸로는 의미를 제대로 담을 수 없다.
#그렇기 때문에 서열을 나눈다던지 누가 더 가치가 있다는지에 대해 측정할 수 없다.
#이를 해결하기 위해 원-핫 인코딩(One-Hot Encoding)을 사용한다.
# 1. 중복을 제거한 문자들을 대상으로 고유 번호를 매긴다. (정수 인코딩) // 범주형 변수를 숫자로 치환해주는 것과 같다.
# 2. 이를 기반으로 희소 벡터를 생성한다.
#희소 벡터 값을 만든다. (표현하고자 하는 건 1 나머지는 0 으로 만든다)
#EX) [감자, 고구가, 피망, 사과, 딸기] = [0,1,2,3,4]
#      피망 = [0,0,1,0,0]
#물론 원 핫 인코딩은 크기가 크면 벡터 저장 공간이 커진다.
# 즉 단어가 1000개 라면 1000개에 해당하는 벡터의 크기가 입력 되어야 된다.
# 단어를 단순하게 숫자로 바꾸기 때문에 의미간의 유사도는 표현 못하는 단점이 있다.

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

#1043 rows x 8 columns
#print(DF_Hand)
#print("=======")


one_hot_Encoding(DF_Hand, 'Pclass')
one_hot_Encoding(DF_Hand, 'Sex')
one_hot_Encoding(DF_Hand, 'Embarked')

#1043rows x 13 columns 별다른 지정을 하지 않았지만 각 문자열마다 1,0 또는 3개인 Embarked의 경우 1,0,0 으로 columns의 값이 늘어났다.
#print(DF_Hand)

y_test, y_train = DF_Hand["Survived"][:300].to_numpy(), DF_Hand["Survived"][300:].to_numpy()

del(DF_Hand["Survived"])
X_test, X_train = DF_Hand[:300].values, DF_Hand[300:].values



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


#모델 학습 relu 와 sigmoid는 상황에 따라서 linear 을 사용 할 수도 있다. 자료의 형질에 따라 프로그래머가 선택해야 된다.
X_test, X_train = scale_adjust(X_test, X_train, 0, key="min_max")
X_test, X_train = scale_adjust(X_test, X_train, 3, key="min_max")

model = keras.Sequential()
model.add(Dense(128, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(16, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

opt = keras.optimizers.Adam(learning_rate = 0.005)
model.compile(optimizer = opt,
              loss = "binary_crossentropy",
              metrics=["binary_accuracy"])

model.fit(X_train, y_train, epochs = 500)

pred = model.predict(X_test).reshape(X_test.shape[0])
pred = np.where(pred > 0.5, 1, 0)
accuracy = 1 - (np.where((pred - y_test) == 0, 0, 1).sum()/len(y_test))

#78%의 성능으로 오히려 Titanic에서 0,1 로만 작성해서 했던거와 유사한 성능을 보인다.
print("Accuracy:", accuracy)



