from tensorflow import keras

#데이터 로드와 호출
Mnist_model = keras.models.load_model("./Model/data/MNIST_202108.h5")

print(Mnist_model.summary())