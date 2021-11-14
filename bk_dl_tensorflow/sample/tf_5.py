import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import numpy as np

tf.random.set_seed(0)


# 1. 훈련 데이터 준비하기
x_train = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_train = [[0], [0], [0], [1]]


# 2. 모델 구성하기
model = keras.Sequential([
    keras.layers.Dense(units=3, input_shape=[2], activation='relu'),
    # keras.layers.Dense(units=3, input_shape=[2], activation='sigmoid'),
    keras.layers.Dense(units=1)
])


# 3. 모델 컴파일하기
# model.compile(loss='mse', optimizer='SGD')
model.compile(loss='mse', optimizer='Adam')


# 4. 모델 훈련하기
pred_before_training = model.predict(x_train)
print('Before Training: \n', pred_before_training)

history = model.fit(x_train, y_train, epochs=1000, verbose=0)

pred_after_training = model.predict(x_train)
print('After Training: \n', pred_after_training)


# 5. 손실값 확인하기

loss = history.history['loss']
plt.plot(loss)
plt.xlabel('Epoch', labelpad=15)
plt.ylabel('Loss', labelpad=15)

plt.show()
