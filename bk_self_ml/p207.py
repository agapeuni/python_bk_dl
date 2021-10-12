 # 확률적 경사 하강법

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

fish = pd.read_csv('bk_self_ml/fish.csv')
print("head :", fish.head())

fish_input = fish[['Weight', 'Length',
                   'Diagonal', 'Height', 'Width']].to_numpy()
fish_target = fish['Species'].to_numpy()

# 훈련 데이터, 테스트 데이터
train_input, test_input, train_target, test_target = train_test_split(
    fish_input, fish_target, random_state=42)

# 표준화 전처리
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)

sc = SGDClassifier(loss='log', max_iter=10, random_state=42)
sc.fit(train_scaled, train_target)
print("1 train :", sc.score(train_scaled, train_target))
print("1 test :", sc.score(test_scaled, test_target))

sc.partial_fit(train_scaled, train_target)
print("2 train :", sc.score(train_scaled, train_target))
print("2 test :", sc.score(test_scaled, test_target))

# 에포크와 과대/과소적합

sc = SGDClassifier(loss='log', random_state=42)
train_score = []
test_score = []

classes = np.unique(train_target)
for _ in range(0, 300):
    sc.partial_fit(train_scaled, train_target, classes=classes)

    train_score.append(sc.score(train_scaled, train_target))
    test_score.append(sc.score(test_scaled, test_target))

plt.plot(train_score)
plt.plot(test_score)
plt.xlabel('epoch')
plt.ylabel('accuracy')
#plt.show()

# max_iter = 100
sc = SGDClassifier(loss='log', max_iter=100, tol=None, random_state=42)
sc.fit(train_scaled, train_target)
print("3 train :", sc.score(train_scaled, train_target))
print("3 test :", sc.score(test_scaled, test_target))

# logg = 'hinge'
sc = SGDClassifier(loss='hinge', max_iter=100, tol=None, random_state=42)
sc.fit(train_scaled, train_target)

print("4 train :", sc.score(train_scaled, train_target))
print("4 test :", sc.score(test_scaled, test_target))
