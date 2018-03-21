#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TensorFlow - stretching our legs 
"""

import tensorflow as tf

a = tf.constant( [10] )
b = tf.constant( [20] )
c = tf.add( a, b )

with tf.Session() as sess:
    result = sess.run( c )
    print( 'output: ', result )
    
sess.close()