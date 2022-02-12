# 1. 라이브러리 읽어 들이기
import tensorflow as tf
import tflearn
import tflearn.datasets.mnist as mnist

from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np


# 2. 데이터 전처리
# MNIST 데이터를 ./data/mnist에 내려받고, 압축을 해제한 다음 각 변수에 할당하기
trainX, trainY, testX, testY = mnist.load_data('./data/mnist/', one_hot=True)


# 3. 모델 생성

# 초기화하기
# tf.reset_default_graph()

# 입력 레이어 만들기
net = tflearn.input_data(shape=[None, 784])

# 중간 레이어 만들기
net = tflearn.fully_connected(net, 128, activation='relu')
net = tflearn.dropout(net, 0.5)

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
