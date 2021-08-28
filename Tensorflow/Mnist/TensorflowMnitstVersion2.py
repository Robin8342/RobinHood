import tensorflow as tf
from tensorflow import keras
import os
#tensorflow 2.x에선 keras에 datasets이 제공한다.

""" 가중치 이름이 다르면 저장 후에 사용할 때도 이 이름으로 사용해야 된다.
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
"""

"""
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])
"""

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0

def create_model():
  model = tf.keras.models.Sequential([
    keras.layers.Dense(512, activation='relu', input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10)
  ])

  model.compile(optimizer='adam',
                loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

  return model

# 모델 객체를 만듭니다
model = create_model()

# 모델 구조를 출력합니다
#model.summary()

#훈련된 모델을 다시 훈련할 필요 없이 사용하거나 훈련 과정이 중단된 경우 훈련을 다시 시작하기 위한 방식
checkpoint_path = "training_1/cp.ckpt"
#dir name setting
checkpoint_dir = os.path.dirname(checkpoint_path)

# 모델의 가중치를 저장하는 콜백 만들기
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

# 새로운 콜백으로 모델 훈련하기
#마찬가지로 제일 위의 가중치 이름과 동일해야된다. train_images, train_labels
model.fit(train_images,
          train_labels,
          epochs=10,
          validation_data=(test_images,test_labels),
          callbacks=[cp_callback])

"""
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test,  y_test, verbose=2)
"""

os.listdir(checkpoint_dir)

#model.save('./saved_model')

#model은 hdf5 표준으로 따르는 기본 저장 포맷도 제공한다.
model.save('save_name.h5')