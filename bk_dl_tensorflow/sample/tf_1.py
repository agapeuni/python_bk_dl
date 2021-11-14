import tensorflow as tf
from tensorflow import keras
import numpy as np

model = keras.Sequential([keras.layers.Dense(units=3, input_shape=[1])])
model.compile(loss='mse')

pred = model.predict([0])
print(pred)

model.evaluate([0], [[0, 1, 0]])
# 1/1 [==============================] - 0s 33ms/sample - loss: 0.3333