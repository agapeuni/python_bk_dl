import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def get_one_result(idx):
    img, y_true, y_pred, confidence = x_test[idx], y_test[idx], np.argmax(
        preds[idx]), 100*np.max(preds[idx])
    return img, y_true, y_pred, confidence


# MNIST 데이터셋을 로드
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 데이터셋 확인
print('x_train.shape =', x_train.shape)
print('y_train.shape =', y_train.shape)
print('x_test.shape =', x_test.shape)
print('y_test.shape =', y_test.shape)

# MNIST 데이터셋 이미지를 표시
fig, axes = plt.subplots(4, 5)
fig.set_size_inches(8, 5)

for i in range(20):
    ax = axes[i//5, i % 5]
    ax.imshow(x_train[i])
    ax.axis('off')
    ax.set_title(str(y_train[i]))

plt.tight_layout()
plt.show()

# 데이터 정규화
x_train = x_train / 255
x_test = x_test / 255

# 모델 생성
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax'),
])

# 모델 컴파일
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 훈련
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10,)

# 평기
test_loss, test_acc = model.evaluate(x_test,  y_test)
print("test_loss =", test_loss)
print("test_acc =", test_acc)

# 예측
preds = model.predict(x_test)
print(np.argmax(preds[:20], axis=1))


fig, axes = plt.subplots(4, 5)
fig.set_size_inches(12, 10)

# 데이터 시각화
for i in range(20):
    ax = axes[i//5, i % 5]
    img, y_true, y_pred, confidence = get_one_result(i)
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'True: {y_true}')
    ax.set_xlabel(f'Prediction: {y_pred}\nConfidence: ({confidence:.2f} %)')

plt.tight_layout()
plt.show()
