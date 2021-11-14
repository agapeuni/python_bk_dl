# 라이브러리 불러오기
import tensorflow as tf

# 2차원 배열 정의
list_of_list = [[10, 20], [30, 40]]
# 텐서 변환 - constant 함수에 2차원 배열 입력
mat1 = tf.constant(list_of_list)
print("rank:", tf.rank(mat1))
print("mat1:", mat1)

# 1차원 벡터 정의
vec1 = tf.constant([1, 0])
vec2 = tf.constant([-1, 2])
# 텐서 변환 - stack 함수로 1차원 배열을 위아래로 쌓기
mat2 = tf.stack([vec1, vec2])
print("rank:", tf.rank(mat2))
print("mat2:", mat2)

# 행렬곱 연산
mat_mul = tf.matmul(mat1, mat2)
print("result:", mat_mul)
print("rank:", tf.rank(mat_mul))

# 덧셈 연산
add1 = tf.math.add(mat1, mat2)
print("result:", add1)
print("rank:", tf.rank(add1))

# 덧셈 연산자
add2 = mat1 + mat2
print("result:", add2)
print("rank:", tf.rank(add2))

# 텐서를 넘파이로 변환
np_arr = mat_mul.numpy()
print(type(np_arr))
print(np_arr)
