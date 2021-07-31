from cycler import cycler
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# 모든 마크 케이스 및 색상 케이스 목록을 정의
cases = [None,
         8,
         (30, 8),
         [16, 24, 30],
         [0, -1],
         slice(100, 200, 3),
         0.1,
         0.3,
         1.5,
         (0.0, 0.1),
         (0.45, 0.1)]

colors = ['#1f77b4',
          '#ff7f0e',
          '#2ca02c',
          '#d62728',
          '#9467bd',
          '#8c564b',
          '#e377c2',
          '#7f7f7f',
          '#bcbd22',
          '#17becf',
          '#1a55FF']

# 케이스와 색상을 동시에 순환하도록 구성
mpl.rcParams['axes.prop_cycle'] = cycler(markevery=cases, color=colors)

# 데이터 포인트 및 오프셋 생성
x = np.linspace(0, 2 * np.pi)
offsets = np.linspace(0, 2 * np.pi, 11, endpoint=False)
yy = np.transpose([np.sin(x + phi) for phi in offsets])

# 마커와 제목으로 플롯 곡선 설정
fig = plt.figure()
ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])

for i in range(len(cases)):
    ax.plot(yy[:, i], marker='o', label=str(cases[i]))
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

plt.title('markevery')
plt.show()