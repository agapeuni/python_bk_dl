
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# 도미 길이
bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 31.5, 32.0, 32.0, 32.0,
                33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0]
# 도미 무게
bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 500.0, 340.0, 600.0, 600.0, 700.0,
                700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0]

# 빙어 길이
smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3,
                11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
# 빙어 무게
smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7,
                10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]


# 도미와 빙어를 그래프로 표시
plt.scatter(bream_length, bream_weight)
plt.scatter(smelt_length, smelt_weight)
plt.xlabel('length')
plt.ylabel('weight')
plt.show()


# 하나의 리스트로 병합
length = bream_length + smelt_length
weight = bream_weight + smelt_weight

# 2차원 리스트로 변환
fish_data = [[l, w] for l, w in zip(length, weight)]
print("fish_data =", fish_data)

# 정답 데이터 생성 (1:도미, 0:빙어)
fish_target = [1]*35 + [0]*14
print("fish_target =", fish_target)

########################
# K 최근접 이웃 알고리즘
kn = KNeighborsClassifier()

# 훈련
kn.fit(fish_data, fish_target)

# 평가
s = kn.score(fish_data, fish_target)
print("score =", s)

plt.scatter(bream_length, bream_weight)
plt.scatter(smelt_length, smelt_weight)
plt.scatter(30, 600, marker='^')
plt.scatter(20, 30, marker='+')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

# 예측 (1: 도미, 0: 빙어)
p = kn.predict([[30, 600], [20, 30]])
print("predict =", p)

# fish_data
print("kn._fit_X =", kn._fit_X)

# fish_target
print("kn._y = ", kn._y)
