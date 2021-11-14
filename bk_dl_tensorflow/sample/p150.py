import tensorflow as tf
import numpy as np

# MNIST 데이터 준비
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 데이터 전처리
x_train = x_train / 255
x_test = x_test / 255

# 모델 생성
model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# 모델 컴파일
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 모델 요약
print(model.summary())
print()

# 학습
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10)

# 평가
test_loss, test_acc = model.evaluate(x_test,  y_test)

print('test_loss =', test_loss)
print('test_acc =', test_acc)
print()

#  예측
preds = model.predict(x_test)
print("y_test =", y_test[:20])
print("preds = ", np.argmax(preds[:20], axis=1))
