# 라이브러리 불러오기
import tensorflow as tf
import numpy as np

# 2차원 배열 정의
mat1 = [[1, 2, 3, 4], 
        [5, 6, 7, 8]]
 
mat2 = [[9, 10, 11, 12],
        [13, 14, 15, 16]]   

mat3 = [[17, 18, 19, 20],
        [21, 22, 23, 24]]    

# 텐서 변환 - constant 함수에 3차원 배열 입력
tensor1 = tf.constant([mat1, mat2, mat3])

# 랭크 확인
print("rank1:", tf.rank(tensor1))

# 텐서 출력
print("tensor1:", tensor1)

# 텐서 변환 - stack 함수로 2차원 배열을 위아래로 쌓기
tensor2 = tf.stack([mat1, mat2, mat3])

# 랭크 확인
print("rank2:", tf.rank(tensor2))

# 텐서 출력
print("tensor2:", tensor2)

# 1차원 배열 정의
vec1 = [1, 2, 3, 4] 
vec2 = [5, 6, 7, 8] 
vec3 = [9, 10, 11, 12]  
vec4 = [13, 14, 15, 16] 
vec5 = [17, 18, 19, 20] 
vec6 = [21, 22, 23, 24] 

# 1차원 배열을 원소로 갖는 2차원 배열 정의
arr = [[vec1, vec2],
        [vec3, vec4],
        [vec5, vec6]]

# 텐서 변환
tensor3 = tf.constant(arr)

# 랭크 확인
print("rank3:", tf.rank(tensor3))

# 텐서 출력
print("tensor3:", tensor3)

# 랭크-4 텐서 만들기
tensor4 = tf.stack([tensor1, tensor2])

# 랭크 확인
print("rank4:", tf.rank(tensor4))

# 텐서 출력
print("tensor4:", tensor4)