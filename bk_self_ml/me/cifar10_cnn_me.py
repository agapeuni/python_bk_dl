
from tensorflow import keras
from keras.datasets import cifar10
import numpy as np

# 데이터셋 로드
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 데이터 정규화
x_train = x_train / 255.0
x_test = x_test / 255.0

# 모델 생성
model = keras.Sequential()
model.add(keras.layers.Conv2D(32, kernel_size=(3, 3),
                              activation='relu', input_shape=(32, 32, 3)))
model.add(keras.layers.MaxPooling2D(2, 2))
model.add(keras.layers.Dropout(0.2))

model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D(2, 2))
model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(keras.layers.Dropout(0.2))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))


# 모델 요약
model.summary()


# 모델 컴파일
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', metrics='accuracy')

# 모델 훈련
history = model.fit(x_train, y_train, epochs=5)

# 모델 평가
loss, accuracy = model.evaluate(x_test, y_test)
print("loss = ", loss)
print("accuracy = ", accuracy)

# 2개 테스트 이미지
test_batch = x_test[:2]


# 모델 예측
preds = model.predict(test_batch)

# 예측값 출력
print("preds =", preds)
print(np.argmax(preds[0]))
print(np.argmax(preds[1]))
