import tensorflow as tf
from tensorflow import keras

from sklearn.model_selection import train_test_split

# 패션 MNIST
(train_input, train_target), (test_input,
                              test_target) = keras.datasets.fashion_mnist.load_data()

train_scaled = train_input / 255.0
train_scaled = train_scaled.reshape(-1, 28*28)

train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size=0.2, random_state=42)

print(train_scaled.shape, train_target.shape)
print(val_scaled.shape, val_target.shape)

# 밀집층
dense = keras.layers.Dense(10, activation='softmax', input_shape=(784,))
model = keras.Sequential(dense)

print(train_scaled.shape, train_target.shape)
print(val_scaled.shape, val_target.shape)

# 컴파일
model.compile(loss='sparse_categorical_crossentropy', metrics='accuracy')
print(train_target[:10])

# 훈련
model.fit(train_scaled, train_target, epochs=5)

# 평가
model.evaluate(val_scaled, val_target)
