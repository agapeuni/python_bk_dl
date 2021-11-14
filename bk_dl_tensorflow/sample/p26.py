# 라이브러리 불러오기
import tensorflow as tf
import numpy as np

# 1차원 배열 정의
py_list = [10., 20., 30.]  # 파이썬 리스트 활용
num_arr = np.array([10., 10., 10.])  # 넘파이 배열 활용

# 텐서 변환
vec1 = tf.constant(py_list, dtype=tf.float32)
vec2 = tf.constant(num_arr, dtype=tf.float32)

# 덧셈 함수
add1 = tf.math.add(vec1, vec2)
print("result:", add1)
print("rank:", tf.rank(add1))

# 덧셈 연산자
add2 = vec1 + vec2
print("result:", add2)
print("rank:", tf.rank(add2))

# tf.math 모듈 함수
print(tf.math.subtract(vec1, vec2))
print(tf.math.multiply(vec1, vec2))
print(tf.math.divide(vec1, vec2))
print(tf.math.mod(vec1, vec2))
print(tf.math.floordiv(vec1, vec2))

# 파이썬 연산자
print(vec1 - vec2)
print(vec1 * vec2)
print(vec1 / vec2)
print(vec1 % vec2)
print(vec1 // vec2)

# 합계 구하기
print(tf.reduce_sum(vec1))
print(tf.reduce_sum(vec2))
