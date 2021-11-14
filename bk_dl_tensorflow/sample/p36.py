# 텐서플로 불러오기
import tensorflow as tf

# 벡터 정의하기
vec = tf.constant([10, 20, 30, 40, 50])
print(vec)
print(vec[0])
print(vec[-1])
print(vec[:3])

# 행렬 정의하기
mat = tf.constant([[10, 20, 30],
                   [40, 50, 60]])
print()
print(mat[0, 2])
print(mat[0, :])
print(mat[:, 1])
print(mat[:, :])

# 랭크-3 텐서 정의하기
tensor = tf.constant([
    [[10, 20, 30],
     [40, 50, 60]],
    [[-10, -20, -30],
     [-40, -50, -60]],
])
print()
print(tensor)
print(tensor[0, :, :])
print(tensor[:, :2, :2])
