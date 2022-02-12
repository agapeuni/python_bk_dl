# 1. 라이브러리 읽어 들이기

import tensorflow as tf
import tflearn

from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

import tflearn.datasets.mnist as mnist
import numpy as np

# 2. 데이터 전처리
trainX, trainY, testX, testY = mnist.load_data('./data/mnist/', one_hot=True)

# 이미지 픽셀 데이터를 1차원에서 2차원으로 변환하기
trainX = trainX.reshape([-1, 28, 28, 1])
testX = testX.reshape([-1, 28, 28, 1])


## 3. 신경망 만들기 ##

## 초기화
#tf.reset_default_graph()

# 입력 레이어 만들기
net = input_data(shape=[None, 28, 28, 1])

# 중간 레이어 만들기
# 합성곱 레이어 만들기
net = conv_2d(net, 32, 5, activation='relu')
# 풀링 레이어 만들기
net = max_pool_2d(net, 2)
# 합성곱 레이어 만들기
net = conv_2d(net, 64, 5, activation='relu')
# 풀링 레이어 만들기
net = max_pool_2d(net, 2)
# 전결합 레이어 만들기
net = fully_connected(net, 128, activation='relu')
net = dropout(net, 0.5)

# 출력 레이어 만들기
net = tflearn.fully_connected(net, 10, activation='softmax')
net = tflearn.regression(net, optimizer='sgd',
                         learning_rate=0.5, loss='categorical_crossentropy')


# 4. 모델 학습
model = tflearn.DNN(net)
model.fit(trainX, trainY, n_epoch=20, batch_size=100,
          validation_set=0.1, show_metric=True)

# 5. 모델 예측
pred = np.array(model.predict(testX)).argmax(axis=1)
print("pred = ", pred)

label = testY.argmax(axis=1)
print("label = ", label)

accuracy = np.mean(pred == label, axis=0)
print("accuracy = ", accuracy)
