import matplotlib.pyplot as plt

plt.title('matplotlib')
plt.plot([2, 4, 5, 8, 10], 'r.--', label='one')
plt.plot([10, 15, 25, 40, 60], 'b^:', label='two')
plt.legend()
plt.show()