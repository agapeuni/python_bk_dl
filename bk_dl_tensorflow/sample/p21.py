# 텐서플로 불러오기
import tensorflow as tf

# 스칼라 정의하기
a = tf.constant(3)
b = tf.constant(4)
print("a:", a)
print("b:", b)

# 랭크 확인하기
print(tf.rank(a))

# 자료형 변환
a = tf.cast(a, tf.float32)
b = tf.cast(b, tf.float32)
print(a.dtype)
print(b.dtype)

# 덧셈
c = tf.math.add(a, b)
print("result:", c)
print("rank:", tf.rank(c))

# 뺄셈
print(tf.math.subtract(a, b))

# 나눗셈
print(tf.math.divide(a, b))

# 나눗셈 (나머지)
print(tf.math.mod(a, b))

# 나눗셈 (몫)
print(tf.math.floordiv(a, b))
