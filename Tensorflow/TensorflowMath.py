"""
tf.add : 덧셈
tf.subtract : 뺄셈
tf.multiply : 곱셈
tf.divide : 나눗셈
tf.pow : n^2 (제곱)
tf.negative : 음수 부호
tf.abs : 절댓값
tf.sign : 부호
tf.round : 반올림
tf.math.ceil : 올림
tf.floor : 내림
tf.math.square : 제곱
tf.math.sqrt : 제곱근
"""

import tensorflow as tf

matrix1 = tf.constant([[1,2],[3,4]])
matrix2 = tf.constant([[2,2],[2,2]])

MatrixLength = tf.constant([[1,2]])
MatrixHeight = tf.constant([[3,],[4]])

add = tf.add(matrix1,matrix2)
sub = tf.subtract(matrix1, matrix2)
div = tf.divide(matrix1, matrix2)
multiply = tf.multiply(matrix1,matrix2)

#3 + 8 = 11
MatrixSum = tf.matmul(MatrixLength,MatrixHeight)

#[3x1, 3x2]   ==  [3,6]
#[4x1, 4x2]   ==  [4,8]
MatrixAll = tf.matmul(MatrixHeight,MatrixLength)

#print(tf.add(2,3)) -> 5  (print 에서 바로 입력 가능)
tf.print(add)

#tf.Sessionas as sess 로 호출하여 사용하는건 Tensorflow 1.x 버전이며 Tensorflow 2.x에선 사용하지 않는다.

tf.print(sub)
tf.print(div)
tf.print(multiply)
tf.print(MatrixSum)
tf.print(MatrixAll)