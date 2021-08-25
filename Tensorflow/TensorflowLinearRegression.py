import tensorflow as tf
import numpy
import matplotlib.pyplot as plt #그래프로 보여줌

learning_rate = 0.01

training_epochs = 1000   #전체 반복하는 훈련 횟수 보통 1000번

display_step = 50 #훈련 도중 결과로 비용과 모델의 파라미터를 출력할 간격

#훈련 데이터
train_X = numpy.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
                         7.042,10.791,5.313,7.997,5.654,9.27,3.1])

train_Y = numpy.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
                         2.827,3.465,1.65,2.904,2.42,2.94,1.3])

