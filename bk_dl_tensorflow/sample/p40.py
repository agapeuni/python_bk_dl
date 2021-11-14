# 텐서플로 불러오기
import tensorflow as tf

# 랭크-1 텐서 정의하기
tensor = tf.constant(range(0, 24))
print(tensor)

tensor1 = tf.reshape(tensor, [3, 8])
print(tensor1)

tensor2 = tf.reshape(tensor1, [-1, 4])
print(tensor2)

tensor3 = tf.reshape(tensor2, [-1])
print(tensor3)

tensor4 = tf.reshape(tensor3, [-1, 3, 4])
print(tensor4)

tensor5 = tf.reshape(tensor4, [3, 2, 4])
print(tensor5)

tensor6 = tf.reshape(tensor5, [3, 2, 2, 2])
print(tensor6)
