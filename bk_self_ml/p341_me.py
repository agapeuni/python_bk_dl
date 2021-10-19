import tensorflow as tf
from keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
import numpy as np

# Fashion MNIST 데이터셋 로드
# x_train, x_test: 그레이 스케일 이미지 데이터의 uint8 배열.
# y_train, y_test: 숫자 라벨의 uint8 배열.
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# 훈련 이미지와 레이블 60,000개
print(x_train.shape, y_train.shape)
# 테스트 이미지와 레이블 10,000개
print(x_test.shape, y_test.shape)

"""
# 5개 훈련 이미지 표시
fig, axs = plt.subplots(1, 5, figsize=(15, 15))
for i in range(5):
    axs[i].imshow(x_train[i])
    axs[i].axis('off')
plt.show()

# 5개 훈련 레이블 표시
print([y_train[i] for i in range(5)])
"""

# 데이터 전처리
x_train, x_test = x_train / 255.0, x_test / 255.0


# 모델 생성
# Flatten 클래스는 입력 데이터를 1차원으로 변환한다.
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(512, activation='relu'))
# model.add(tf.keras.layers.Dense(256, activation='relu'))
# model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))


# 모델 요약
model.summary()


# 모델 컴파일
# 옵티마이저(optimizer)는 adam을 사용
# 손실 함수(loss function)는 sparse_categorical_crossentropy를 사용 (다중 분류)
# 지표(metrics)는 accuracy를 지정 (훈련과 테스트를 평가)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


class StopTraining(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
      if logs.get('loss') < 0.3:
          print('stop training.')
          self.model.stop_training = True

callbacks = StopTraining()

# 모델 훈련
# 에포크(epochs)는 전체 이미지 학습 횟수 설정
history = model.fit(x_train, y_train, epochs=10, callbacks=[callbacks])

plt.plot(history.history['loss'])
plt.plot(history.history['accuracy'])
plt.xlabel('epoch')
plt.ylabel('val')
plt.legend(['loss', 'accuracy'])
plt.show()

# 모델 평가
loss, accuracy= model.evaluate(x_test, y_test)
print("loss = ", loss)
print("accuracy = ", accuracy)



# 모델 예측
preds = model.predict(x_test)
# 예측값 출력
print(preds[0])
# 가장 높은 값을 갖는 인덱스를 확인
print(np.argmax(preds[0]))
