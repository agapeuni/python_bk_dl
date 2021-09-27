# 로지스틱 회귀
from scipy.special import softmax
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

fish = pd.read_csv('https://bit.ly/fish_csv_data')
print("head :", fish.head())

# 고유값 추출
print("unique :", pd.unique(fish['Species']))

# 원하는 열을 선택 --> 리스트
fish_input = fish[['Weight', 'Length',
                   'Diagonal', 'Height', 'Width']].to_numpy()
print("fish_input = ", fish_input[:5])

# 타깃 데이터
fish_target = fish['Species'].to_numpy()
print("fish_target = ", fish_target[:5])

# 훈련 데이터, 테스트 데이터
train_input, test_input, train_target, test_target = train_test_split(
    fish_input, fish_target, random_state=42)

# 표준화 전처리
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)


# k-최근접 이웃 분류기의 확률 예측
kn = KNeighborsClassifier(n_neighbors=3)
kn.fit(train_scaled, train_target)

print("train :", kn.score(train_scaled, train_target))
print("test :", kn.score(test_scaled, test_target))

# 정렬된 타깃값
print(kn.classes_)
# 예측
print(kn.predict(test_scaled[:5]))

# 클래스별 확률값
proba = kn.predict_proba(test_scaled[:5])
print(np.round(proba, decimals=4))

# 가까운 이웃 확인
distances, indexes = kn.kneighbors(test_scaled[3:4])
print(train_target[indexes])

"""
# 시그모이드 함수
import numpy as np
import matplotlib.pyplot as plt

z = np.arange(-5, 5, 0.1)
phi = 1 / (1 + np.exp(-z))

plt.plot(z, phi)
plt.xlabel('z')
plt.ylabel('phi')
plt.show()
"""

# 불리언 인덱싱
char_arr = np.array(['A', 'B', 'C', 'D', 'E'])
print(char_arr[[True, False, True, False, False]])

# 도미와 빙어에 대한 데이터 준비
bream_smelt_indexes = (train_target == 'Bream') | (train_target == 'Smelt')

train_bream_smelt = train_scaled[bream_smelt_indexes]
target_bream_smelt = train_target[bream_smelt_indexes]


# 로지스틱 회귀로 이진 분류 수행하기
lr = LogisticRegression()
lr.fit(train_bream_smelt, target_bream_smelt)

# 예측
print(lr.predict(train_bream_smelt[:5]))
# 예측 확률
print(lr.predict_proba(train_bream_smelt[:5]))
# 클래스
print(lr.classes_)
# 학습한 계수
print(lr.coef_, lr.intercept_)

# 처음 5개의 샘플의 z값
decisions = lr.decision_function(train_bream_smelt[:5])
print(decisions)
print(expit(decisions))


# 로지스틱 회귀로 다중 분류 수행하기
lr = LogisticRegression(C=20, max_iter=1000)
lr.fit(train_scaled, train_target)
print("train :", lr.score(train_scaled, train_target))
print("test :", lr.score(test_scaled, test_target))

# 예측
print(lr.predict(test_scaled[:5]))
# 예측 확률
proba = lr.predict_proba(test_scaled[:5])
print(np.round(proba, decimals=3))

# 클래스
print(lr.classes_)
# 다중분류 크기 출력
print(lr.coef_.shape, lr.intercept_.shape)

decision = lr.decision_function(test_scaled[:5])
print(np.round(decision, decimals=2))

# 소프트맥스 함수
proba = softmax(decision, axis=1)
print(np.round(proba, decimals=3))
