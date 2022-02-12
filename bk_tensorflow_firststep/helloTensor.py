import tensorflow as tf

a = tf.constant(2)
b = tf.constant(3)
c = tf.add(a, b)
print(c)

d = tf.add(2, 3).numpy()
print(d)

