#호출은 되는데 저장해서 불러와 테스트 이미지가 안됨. 수정할 것 

import tensorflow as tf
from tensorflow import keras

(test_images, test_labels) = tf.keras.datasets.mnist.load_data()
test_labels = test_labels[:1000]
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0

#개인이 저장한 모델을 호출해서 사용
#그래야 매번 학습하지 않고 바로 사용할 수 있다.
#new_model = tf.keras.models.load_model('saved_model')
new_model = tf.keras.models.load_model('save_name.h5')

#모델 구조 확인
#Mnist param 401920
new_model.summary()

loss, acc = new_model.evaluate(test_images,  test_labels, verbose=2)
print('복원된 모델의 정확도: {:5.2f}%'.format(100*acc))
