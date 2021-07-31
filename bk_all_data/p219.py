import matplotlib.pyplot as plt
import numpy as np

a = np.linspace(1, 3, 33)
print(a)

x = np.random.randint(-100, 100, 1000)  # 1000개의 랜덤 값 추출
y = np.random.randint(-100, 100, 1000)  # 1000개의 랜덤 값 추출
size = np.random.randint(-100, 100, 1000)
mask1 = abs(x) > 50  # x에 저장된 값 중 절댓값이 50보다 큰 값 걸러 냄
mask2 = abs(y) > 50  # y에 저장된 값 중 절댓값이 50보다 큰 값 걸러 냄
x = x[mask1 + mask2]  # mask1과 mask2 중 하나라도 만족하는 값 저장
y = y[mask1 + mask2]  # mask1과 mask2 중 하나라도 만족하는 값 저장
plt.scatter(x, y, s=size, c=x, cmap='jet', alpha=0.7)
plt.colorbar()
plt.show()

