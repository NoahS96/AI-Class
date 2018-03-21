#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hello, TensorFlow!

@author: amandafernandez
"""

import tensorflow as tf

hello = tf.constant('hello, TF!')

session = tf.Session()
print( session.run(hello).decode() )
session.close()