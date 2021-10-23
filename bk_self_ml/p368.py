import tensorflow as tf
from tensorflow import keras

from sklearn.model_selection import train_test_split

(train_input, train_target), (test_input,
                              test_target) = keras.datasets.fashion_mnist.load_data()

train_scaled = train_input / 255.0
train_scaled = train_scaled.reshape(-1, 28*28)

train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size=0.2, random_state=42)

# 심층 신경망 
model = keras.Sequential()
model.add(keras.layers.Dense(100, activation="sigmoid",
                             input_shape=(784, ), name="hidden"))
model.add(keras.layers.Dense(10, activation="softmax", name="output"))

# 모델 요약
model.summary()

# 컴파일
model.compile(loss='sparse_categorical_crossentropy', metrics='accuracy')

# 훈련
model.fit(train_scaled, train_target, epochs=5)

# 평가 
model.evaluate(val_scaled, val_target)
