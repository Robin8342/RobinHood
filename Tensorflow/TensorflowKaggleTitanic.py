#주의점 머신러닝에는 문자가 들어 갈 수 없으므로 모든 문자를 숫자로 바꾸어야 함.
#ex) sex : male = 0, female = 1
#    Embarked: C = 0, Q = 1, S =
import pandas as pd
import numpy as np
import os
from tensorflow import keras
from tensorflow.keras.layers import Dense
from copy import copy


#기존에 test, train, gender_submission.csv의 파일을 받은 경로를 지정해준다.
#데이터 프레임을 딕셔너리로 관리할때 데이터를 가지고 올때 손쉽다. (한 번에 다 가져온다.)
file_path = "KaggleImage/titanic"

def import_Data(file_path):
    result = dict()
    for file in os.listdir(file_path):
        file_name = file[:-4]
        result[file_name] = pd.read_csv(file_path + "/" + file)

    return result

Rawdata_dict = import_Data(file_path)


#pd.merge() = 두 dataFrame을 동일한 Column(열기준)으로 하나로 합친다.
#pd.concat() = 모든 Column이 동일한 두 DataFrame을 행 기준으로 합친다.
#Dataname.reset_index() = [Dataname]의 index를 초기화 한다.
def make_Rawdata(dict_data):
    dict_key = list(dict_data.keys())
    test_Dataset = pd.merge(dict_data["gender_submission"], dict_data["test"], how='outer', on='PassengerId')
    Rawdata = pd.concat([dict_data["train"], test_Dataset])
    Rawdata.reset_index(drop=True, inplace=True)

    return Rawdata

Rawdata = make_Rawdata(Rawdata_dict)

#Dataset 확인
#1309 rows x 12 columns
#print(Rawdata)

def remove_columns(DF, remove_list):
    #원본 정보를 직접적으로 고치지 않기 위해 따로 뺀다.
    result = copy(Rawdata)
    #PassengerId를 기준으로 설정 (Dataname.set_index() = index 설정
    result.set_index("PassengerId", inplace = True)

    #불필요한 column 제거
    for column in remove_list:
        del(result[column])

    return result

remove_list = ["Name", "Ticket"]
DF_Hand1 = remove_columns(Rawdata, remove_list)

#1309 rows x 9columns
#print(DF_Hand1)

#추가로 불필요한 데이터를 삭제하기 위해서 결과값을 살펴보면 cabin(객실번호)는 불필요하게 데이터 값이 높다.
#물론 객실 번호가 배에서 탈출하기 위한 위치에 영향을 줄 수는 있지만 티겟 등급이나 요금 등에서 중복되니 제거해도 된다.
#물론 이는 좋은 데이터 결과를 만들기 위해서 프로그래머가 고민해서 선정하면 된다.
#print(DF_Hand1.isnull().sum())

def missing_value(DF):
    del(DF["Cabin"])
    #Cabin의 모든 행을 제거 한다.
    DF.dropna(inplace = True)

#1043 rows x 8 columns
missing_value(DF_Hand1)
#print(DF_Hand1)


#sex, Embarked의 문자값을 모두 숫자 데이터로 바꾼다.
DF_Hand1["Sex"] = np.where(DF_Hand1["Sex"].to_numpy() == "male", 0, 1)
DF_Hand1["Embarked"] = np.where(DF_Hand1["Embarked"].to_numpy() == "C", 0,
                                np.where(DF_Hand1["Embarked"].to_numpy() == "Q", 1,2))

#print(DF_Hand1)

#데이터를 모두 숫자로 바꾸었으니 이제 Train과 test, label dataset으로 분리 한다.
#train:test 는 기본적으로 7:3을 기준으로 한다.

y_test, y_train = DF_Hand1["Survived"][:300].to_numpy(), DF_Hand1["Survived"][300:].to_numpy()

#서바이벌 열을 제외하고 데이터를 뽑아낸다.
del(DF_Hand1["Survived"])
X_test, X_train = DF_Hand1[:300].values, DF_Hand1[300:].values

#print(X_test)

#나이가 제각각이니 나이의 최소치와 평균치를 구한다.
#fare은 승객 요금으로 제거 되었던 객실 번호처럼 탈출하기 좋은 위치에 영향을 끼치는 요소이다.
#배열내에 [:,] 은 는 소수점 자리
age_min = np.min(X_test[:,2])
age_max = np.max(X_test[:,2])

Fare_min = np.min(X_test[:,5])
Fare_max = np.max(X_test[:,5])

"""
#X_train[:2] 과 X_train[:,2]의 차이 => 행과 열의 출력
print(X_train)
print("==============")
print(X_train[:2])
print("=================")
print(X_train[:,2])
"""


#리스트에서 [:,2]의 경우 열의 정보들을 모두 나열한다.
X_train[:,2] = (X_train[:,2] - age_min)/(age_max - age_min)
X_test[:,2] = (X_test[:,2] - age_min)/(age_max - age_min)

X_train[:,5] = (X_train[:,5] - Fare_min)/(Fare_max - Fare_min)
X_test[:,5] = (X_test[:,5] - Fare_min)/(Fare_max - Fare_min)



#모델 생성
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

#손실 값은 상대적인 값이므로 단순하게 접근해서는 안된다.
#모델을 평가하는 기준 정확도는 실제 분류와 예측한 분류가 얼마나 일치하는지를 보면 된다.

#79%성능값이다. 케글에 보면 100%도 많이 있다.
pred = model.predict(X_test).reshape(X_test.shape[0])
pred = np.where(pred > 0.5, 1, 0)
accuracy = 1 - (np.where((pred - y_test) == 0, 0, 1).sum()/len(y_test))
print("Accuracy:", accuracy)



"""
#데이터 내부 확인  Rawdata_dict -> 총 배열 3
dict_key = list(Rawdata_dict.keys())
print(len(Rawdata_dict))
#print(dict_key)
print(Rawdata_dict[dict_key[2]])
#Rewdata_dict[0] : 고객 번호 , Survied : 생존
#Rewdata_dict[1] : cabin : 객실번호, embarked : 기항지 위치
"""

