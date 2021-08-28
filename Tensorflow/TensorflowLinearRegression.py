import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import random
#2차원에서의 직선 -> 기울기와 y절편을 가지는 좌표평면 위 점들의 집합.
#tensorflow 2.x 부턴 placeholder와 session의 삭제 되어



#극단적 데이터를 제거한 상태에서 선형 회귀선 데이터 값
population_inc = [0.3, -0.78, 1.26, 0.03, 1.11, 0.24, -0.24, -0.47, -0.77, -0.37, -0.85, -0.41, -0.27, 0.02, -0.76, 2.66]
population_old = [12.27, 14.44, 11.87, 18.75, 17.52, 16.37, 19.78, 19.51, 12.65, 14.74, 10.72, 21.94, 12.83, 15.51, 17.14, 14.42]

a = tf.Variable(random.random())
b = tf.Variable(random.random())


#Linear Regression 계산
#제곱을 하는 이유 : 음수와 양수 모두가 나올 수 있으므로 제곱근을 취하는 방법 사용.
def compute_loss():
    """
    print("random variable 하기 전")
    print(population_inc)
    print("==============")
    """
    y_pred = a * population_inc + b
    """
    print("가중치 적용 후")
    print(y_pred)
    print("=====")
    """
    loss = tf.reduce_mean((population_old - y_pred) ** 2)
    """
    print("로스율")
    print(loss)
    """
    return loss


#optimizer(최적화 함수) - 미분 계산 및 가중치 업데이트를 자동으로 진행시켜줌.
#optimizer 값은 주로 0.1 ~ 0.0001 사이의 값을 사용한다.
#Grid Search(격자탐색 - 가장 높은 성능을 보이는 파라미터 찾는 탐색 방법
#대신 시간이 오래 걸린다.
#random Search(Grid search의 실행시간으로 인해 random search는)
#랜덤하게 숫자를 넣은 뒤 정해진 간격(grid)사이에 위치한 값에서도 확률적으로 탐색하는 방법
#최적 hyperparameter 값을 더 빨리 찾을 수 있다.
#random search가 grid search보다 성능이 더 좋다. (논문으로 검증 됨)
optimizer = tf.optimizers.Adam(lr=0.07)

for i in range(1000):
    #Variable.random 으로 최적의 hyperparameter 를 귀하기 위한 optimizer 값.
    optimizer.minimize(compute_loss, var_list=[a,b])

    if i & 100 == 99:
        print(i, 'a:', a.numpy(), 'b:', b.numpy(), 'loss:', compute_loss().numpy())




#Linear Regression 라인 생성
line_X = np.arange(min(population_inc), max(population_inc), 0.01)
line_Y = a * line_X + b


#출력 화면
plt.plot(line_X, line_Y, 'r-')
plt.plot(population_inc, population_old, 'bo')
plt.xlabel('Population Growth Rate') #x 좌표 라벨명
plt.ylabel('Elderly Population Rate') #y 좌표 라벨 명
plt.show()


"""
#각 데이터들의 평균 값을 구한다.
x_bar = sum(population_inc) / len(population_inc)
y_bar = sum(population_old) / len(population_old)

#최소제곱법으로 a,b의 값을 구한다.
#zip은 zip(x,y)처럼 각각의 값을 데응하여 리스트를 만들어 저장한다.
a = sum([(y - y_bar) * (x - x_bar) for y, x in list(zip(population_old,population_inc))])
a /= sum([(x - x_bar) ** 2 for x in population_inc])
b = y_bar - a * x_bar

print('a:', a, 'b:', b)

#그리프를 그리기 위한 데이터
line_x = np.arange(min(population_inc), max(population_inc), 0.01)
line_y = a * line_x + b

plt.plot(line_x, line_y, 'r-')
plt.plot(population_inc, population_old,'bo')
plt.xlabel('Population Growth Rate') #x 좌표 라벨명
plt.ylabel('Elderly Population Rate') #y 좌표 라벨 명
plt.show()
"""

"""
population_inc = [0.3, -0.78, 1.26, 0.03, 1.11, 15.17, 0.24, -0.24, -0.47, -0.77, -0.37, -0.85, -0.41, -0.27, 0.02, -0.76, 2.66]
population_old = [12.27, 14.44, 11.87, 18.75, 17.52, 9.29, 16.37, 19.78, 19.51, 12.65, 14.74, 10.72, 21.94, 12.83, 15.51, 17.14, 14.42]

#15.17 / 9.29 은 일반적인 경향에서 벗어난 데이터 이므로 극단치는 제거하는 것이 일반적으로 파악하기가 좋다.
population_inc = population_inc[:5] + population_inc [6:]
population_old = population_old[:5] + population_old [6:]

plt.plot(population_inc, population_old,'bo')
plt.xlabel('Population Growth Rate') #x 좌표 라벨명
plt.ylabel('Elderly Population Rate') #y 좌표 라벨 명
plt.show()
"""