import tensorflow as tf
#추가 정보(하나의 파일 v2로 변경) : tf_upgrade_v2 -- infile [filename].py --outfile [filename]-upgraded.py
#폴더 전체 v2로 변경 : tf_upgrade_v2 --intree coolcode --outtree coolcode-upgraded

#Tensorflow 1.x version
"""
in_a = tf.placeholder(dtype=tf.float32, shape=(2))
in_b = tf.placeholder(dtype=tf.float32, shape=(2))

def forward(x):
  with tf.variable_scope("matmul", reuse=tf.AUTO_REUSE):
    W = tf.get_variable("W", initializer=tf.ones(shape=(2,2)),
                        regularizer=tf.contrib.layers.l2_regularizer(0.04))
    b = tf.get_variable("b", initializer=tf.zeros(shape=(2)))
    return W * x + b

out_a = forward(in_a)
out_b = forward(in_b)

reg_loss = tf.losses.get_regularization_loss(scope="matmul")

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  outs = sess.run([out_a, out_b, reg_loss],
                feed_dict={in_a: [1, 0], in_b: [0, 1]})
"""


#placeholder & Session가 tensorflow 2.x에서 삭제 되고 문장이 직관적으로 설정됐다.
#variable에서 곧바로 shape와 이름을 정해 사용할 수 있으며
#def forward 호출에서도 별도의 값을 지정하는게 아닌 수식을 입력하면 된다.

W = tf.Variable(tf.ones(shape=(2,2)), name="W")
b = tf.Variable(tf.zeros(shape=(2)), name="b")

def forward(x):
    return W * x + b


out_a = forward([1,0])
print(out_a)