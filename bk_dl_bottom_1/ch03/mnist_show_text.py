# coding: utf-8
import sys
import os
sys.path.append(os.pardir)

import numpy as np
from dataset.mnist import load_mnist
from PIL import Image


(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

img = x_train[3]
label = t_train[3]

print(label)
i = 1
for x in img:
    print('{:3} '.format(x), end='')
    if i == 28:
        print()
        i = 1
    else:
        i = i + 1
