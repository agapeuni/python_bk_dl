import matplotlib.pyplot as plt
import numpy as np

# 샘플 데이터셋 생성
x = np.arange(1, 6)

# y = 3x + 2
y = 3 * x + 2
print(x)
print(y)


# 시각화
plt.plot(x, y)
plt.title('y = 3x + 2')
plt.show()
