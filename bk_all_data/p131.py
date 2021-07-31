import matplotlib.pyplot as plt

size = [24, 22, 18, 12]
label = ['A', 'B', 'AB', 'O']

plt.rc('font', family = 'Malgun Gothic')            
#plt.axis('equal')
plt.pie(size, labels=label, autopct='%.1f%%', explode=(0,0,0,0.1))
plt.legend()
plt.show()