#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2019. All rights reserved.
Created by C. L. Wang on 2019/9/29
"""
import tensorflow as tf
import numpy as np


def main():
    img_np = np.array([[1, 2], [3, 4]])
    img_x = img_np - [2, 2]
    print(img_x)

    img_tensor = tf.constant(img_np)
    img_tensor -= [2, 2]
    sess = tf.Session()
    x = sess.run(img_tensor)
    print(x)


if __name__ == '__main__':
    main()
