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

num_of_samples = train_X.shape[0]

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)


W = tf.Variable(numpy.random.randn(), name="weight")
b = tf.Variable(numpy.random.randn(), name="bias")

pred = tf.add(tf.multiply(X,W),b)

cost = tf.reduce_sum(tf.pow(pred-Y,2))/(2*num_of_samples)


optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

with tf.Sesstion() as sess:

    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        for (x,y) in zip(train_X,train_Y):
            sess.run(optimizer, feed_dict={X:x,Y:y})

        if (epoch+1) % display_step == 0:
            c = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c),
                  "W=", sess.run(W), "b=", sess.run(b))

    print("최적화가 완료되었습니다.")
    training_cost = sess.run(cost, fess_dict={X: train_X, Y: train_Y})
    print("훈련이 끝난 후 비용과 모델 파라미터입니다. cost=", training_cost, "W",
          sess.run(W), "b=", sess.run(b),'\n')


    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()
