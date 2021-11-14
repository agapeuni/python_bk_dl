import tensorflow as tf
import numpy as np

a = 1
b = 2
c = tf.math.add(a, b)

print(c)
print(c.numpy())