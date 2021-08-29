import pandas as pd
import numpy as np
import os


#기존에 test, train, gender_submission.csv의 파일을 받은 경로를 지정해준다.
#데이터 프레임을 딕셔너리로 관리할때ㅇ 데이터를 가지고 올때 손쉽다. (한 번에 다 가져온다.)
file_path = "KaggleImage/titanic"

def import_Data(file_path):
    result = dict()
    for file in os.listdir(file_path):
        file_name = file[:-4]
        result[file_name] = pd.read_csv(file_path + "/" + file)

    return result

Rawdata_dict = import_Data(file_path)

"""
#데이터 내부 확인  Rawdata_dict -> 총 배열 3
dict_key = list(Rawdata_dict.keys())
print(len(Rawdata_dict))
#print(dict_key)
print(Rawdata_dict[dict_key[2]])
#Rewdata_dict[0] : 고객 번호 , Survied : 생존
#Rewdata_dict[1] : cabin : 객실번호, embarked : 기항지 위치
"""

