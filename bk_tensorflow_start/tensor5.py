## 1. 라이브러리 읽어 들이기 ##

# 텐서플로 라이브러리
import tensorflow as tf
# tflearn 라이브러리
import tflearn
# mnist 데이터 세트를 다루기 위한 라이브러리
import tflearn.datasets.mnist as mnist

import numpy as np

## 2. 데이터 읽어 들이고 전처리하기 ##
# MNIST 데이터를 ./data/mnist에 내려받고, 압축을 해제한 다음 각 변수에 할당하기
trainX, trainY, testX, testY = mnist.load_data('./data/mnist/', one_hot=True)

# 이미지 픽셀 데이터를 1차원에서 시계열 데이터로 변환하기
trainX = np.reshape(trainX, (-1, 28, 28))

## 3. 신경망 만들기 ##

# 초기화
# tf.reset_default_graph()

# 입력 레이어 만들기
net = tflearn.input_data(shape=[None, 28, 28])

# 중간 레이어 만들기
# LSTM 블록
net = tflearn.lstm(net, 128)

# 출력 레이어 만들기
net = tflearn.fully_connected(net, 10, activation='softmax')
net = tflearn.regression(net, optimizer='sgd',
                         learning_rate=0.5, loss='categorical_crossentropy')

## 4. 모델 만들기(학습) ##
# 학습 실행하기
model = tflearn.DNN(net)
model.fit(trainX, trainY, n_epoch=20, batch_size=100,
          validation_set=0.1, show_metric=True)
