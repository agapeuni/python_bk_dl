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
    keras.layers.Dense(units=1)
])

# 3. 모델 컴파일하기
model.compile(loss='mse', optimizer='Adam')

# 4. 모델 훈련하기
pred_before_training = model.predict(x_train)
print('Before Training: \n', pred_before_training)

history = model.fit(x_train, y_train, epochs=100, verbose=1)

pred_after_training = model.predict(x_train)
print('After Training: \n', pred_after_training)


# 5. 손실값 확인하기
import matplotlib.pyplot as plt

loss = history.history['loss']
plt.plot(loss)
plt.xlabel('Epoch', labelpad=15)
plt.ylabel('Loss', labelpad=15)

plt.show()

import matplotlib.pyplot as plt
import numpy as np

plt.style.use('default')
plt.rcParams['figure.figsize'] = (6, 4)
plt.rcParams['font.size'] = 14

plt.plot(pred_before_training, 's-', markersize=10, label='pred_before_training')
plt.plot(pred_after_training, 'd-', markersize=10, label='pred_after_training')
plt.plot(y_train, 'o-', markersize=10, label='y_train')

plt.xticks(np.arange(4), labels=['[0, 0]', '[0, 1]', '[1, 0]', '[1, 1]'])
plt.xlabel('Input (x_train)', labelpad=15)
plt.ylabel('Output (y_train)', labelpad=15)

plt.legend()
plt.show()