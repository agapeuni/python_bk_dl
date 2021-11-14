# 라이브러리 불러오기
import tensorflow as tf
import numpy as np

# 1차원 배열 정의
py_list = [10., 20., 30.]  # 파이썬 리스트 활용
num_arr = np.array([10., 10., 10.])  # 넘파이 배열 활용

# 텐서 변환
vec1 = tf.constant(py_list, dtype=tf.float32)
vec2 = tf.constant(num_arr, dtype=tf.float32)

# 거듭제곱
print(tf.math.square(vec1))

# 거듭제곱 (파이썬 연산자)
print(vec1**2)

# 제곱근
print(tf.math.sqrt(vec2))

# 제곱근 (파이썬 연산자)
print(vec2**0.5)

# 브로드캐스팅 연산
print(vec1 + 1)
print(vec2 + 2)
