import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import numpy as np

tf.random.set_seed(0)

model = keras.Sequential([keras.layers.Dense(units=3, input_shape=[1])])
model.compile(loss='mse', optimizer='SGD')

model.fit([1], [[0, 1, 0]], epochs=1)
model.evaluate([1], [[0, 1, 0]])

history = model.fit([1], [[0, 1, 0]], epochs=100)


plt.style.use('default')
plt.rcParams['figure.figsize'] = (4, 3)
plt.rcParams['font.size'] = 12

loss = history.history['loss']
plt.plot(loss)
plt.xlabel('Epoch')
plt.ylabel('Loss')
# plt.show()


plt.style.use('default')
plt.rcParams['figure.figsize'] = (4, 3)
plt.rcParams['font.size'] = 12

tf.random.set_seed(0)

model = keras.Sequential(
    [keras.layers.Dense(units=3, input_shape=[1], use_bias=False)])
model.compile(loss='mse', optimizer='SGD')

pred = model.predict([1])
print(pred)
print(model.get_weights())

plt.bar(np.arange(3), pred[0])
plt.ylim(-1.1, 1.1)
plt.xlabel('Output Node')
plt.ylabel('Output')
plt.text(-0.4, 0.8, 'Epoch 0')
plt.tight_layout()
plt.savefig('./bk_dl_tensorflow/test/pred000.png')
plt.clf()

epochs = 500
for i in range(1, epochs+1):
    model.fit([1], [[0, 1, 0]], epochs=1, verbose=0)
    pred = model.predict([1])

    if i % 25 == 0:
        plt.bar(np.arange(3), pred[0])
        plt.ylim(-1.1, 1.1)
        plt.xlabel('Output Node')
        plt.ylabel('Output')
        plt.text(-0.4, 0.8, 'Epoch ' + str(i))
        plt.tight_layout()
        plt.savefig('./bk_dl_tensorflow/test/pred' + str(i).zfill(3) + '.png')

        plt.clf()

print(pred)
print(model.get_weights())
