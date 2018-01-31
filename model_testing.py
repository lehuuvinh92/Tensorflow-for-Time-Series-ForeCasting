#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 14:02:54 2018

@author: acer
"""
import tensorflow as tf

# Test model
def test_model(sess, out, x, y, x_test, y_test):
	# Print the final accuracy
	correct_prediction = tf.equal(tf.argmax(out, 1), tf.argmax(y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	print("Test Accuracy: ", (sess.run(accuracy, feed_dict={x: x_test, y: y_test})))
	
	# Print the final mean square error
	pred_y = sess.run(y, feed_dict={x:x_test})
	mse = tf.reduce_mean(tf.square(pred_y - y_test))
	print("MSE: %.4f" % sess.run(mse))

