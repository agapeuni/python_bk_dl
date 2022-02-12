import numpy as np
import datetime
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

n = 10
A = np.random.rand(1000, 1000).astype('float32')
B = np.random.rand(1000, 1000).astype('float32')

c1 = []
c2 = []


def matpow(M, n):
    if n < 1:  # Abstract cases where n < 1
        return M
    else:
        return tf.matmul(M, matpow(M, n-1))


with tf.device('/gpu:0'):
    a = tf.constant(A)
    b = tf.constant(B)
    # compute A^n and B^n and store results in c1
    c1.append(matpow(a, n))
    c1.append(matpow(b, n))

with tf.device('/cpu:0'):
    sum = tf.add_n(c1)  # Addition of all elements in c1, i.e. A^n + B^n

t1_1 = datetime.datetime.now()
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    # Runs the op.
    print(sess.run(sum))

t2_1 = datetime.datetime.now()
