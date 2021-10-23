from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_validate
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

# 패션 MNIST
(train_input, train_target), (test_input,
                              test_target) = keras.datasets.fashion_mnist.load_data()

print(train_input.shape, train_target.shape)
print(test_input.shape, test_target.shape)

# 처음 10개 이미지 표시
fig, axs = plt.subplots(1, 10, figsize=(10, 10))
for i in range(10):
    axs[i].imshow(train_input[i], cmap='gray_r')
    axs[i].axis('off')
plt.show()


print([train_target[i] for i in range(10)])
print(np.unique(train_target, return_counts=True))

# 정규화
train_scaled = train_input / 255.0
train_scaled = train_scaled.reshape(-1, 28*28)
print(train_scaled.shape)

# 로지스틱 회귀
sc = SGDClassifier(loss='log', max_iter=5, random_state=42)
scores = cross_validate(sc, train_scaled, train_target, n_jobs=-1)
print(np.mean(scores['test_score']))
