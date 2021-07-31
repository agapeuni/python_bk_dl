import matplotlib.pyplot as plt
import random

x = []
y = []
s = []

for i in range(100):
    x.append(random.randint(15, 100))
    y.append(random.randint(15, 100))
    s.append(random.randint(10, 100))

'''
plt.scatter([1, 2, 3, 4, 5], [10, 30, 50, 40, 20]
            , s=[100, 200, 300, 400, 500]
            , c=range(5)
            , cmap='jet')
'''
plt.scatter(x, y, s=s, c=s, cmap='jet', alpha=0.5)
plt.colorbar()
plt.plot(range(max(x)), range(max(y)), 'g')
plt.show()
