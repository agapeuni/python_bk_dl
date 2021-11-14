# 라이브러리 불러오기
import tensorflow as tf

# 2차원 배열 정의
list_of_list = [[10, 20], [30, 40]] 
# 텐서 변환 - constant 함수에 2차원 배열 입력
mat1 = tf.constant(list_of_list)
# 랭크 확인
print("rank:", tf.rank(mat1))
# 텐서 출력
print("mat1:", mat1)

# 1차원 벡터 정의
vec1 = tf.constant([1, 0])
vec2 = tf.constant([-1, 2])
# 텐서 변환 - stack 함수로 1차원 배열을 위아래로 쌓기
mat2 = tf.stack([vec1, vec2])
# 랭크 확인
print("rank:", tf.rank(mat2))
# 텐서 출력하기
print("mat2:", mat2)
