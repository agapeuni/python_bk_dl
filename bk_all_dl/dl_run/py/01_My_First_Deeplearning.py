from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import tensorflow as tf
import pandas as pd

np.random.seed(3)
tf.random.set_seed(3)
Data_set = np.loadtxt("bk_dl_all/dataset/ThoraricSurgery.csv", delimiter=",")

# 환자기록
X = Data_set[:, 0:17]
# 수술 결과
Y = Data_set[:, 17]

# 모델설정 및 실행
model = Sequential()
model.add(Dense(100, input_dim=17, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 딥러닝 실행
model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, epochs=10, batch_size=10)
