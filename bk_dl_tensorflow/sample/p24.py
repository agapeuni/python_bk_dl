# 라이브러리 불러오기
import tensorflow as tf
import numpy as np

# 1차원 배열 정의
py_list = [10., 20., 30.]  # 파이썬 리스트 활용
num_arr = np.array([10., 10., 10.])  # 넘파이 배열 활용

# 텐서 변환
vec1 = tf.constant(py_list, dtype=tf.float32)
vec2 = tf.constant(num_arr, dtype=tf.float32)

# 텐처 출력
print("vec1:", vec1)
print("vec2:", vec2)

# 랭크 확인
print(tf.rank(vec1))
print(tf.rank(vec2))
