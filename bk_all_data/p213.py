import matplotlib.pyplot as plt
import numpy as np

x = np.random.randint(10, 100, 200)
y = np.random.randint(10, 100, 200)
size = np.random.randint(10, 100, 200)

plt.scatter(x, y, s=size, c=x, cmap='jet', alpha=0.7)
plt.colorbar()
plt.show()
