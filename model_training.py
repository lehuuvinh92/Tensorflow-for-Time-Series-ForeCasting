# -*- coding: utf-8 -*-
# Created on Fri Jan 12 18:39:59 2018
# @author: acer
# =====================================

"""Create a training model."""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Define the important parameters and variable to work with the tensors
learning_rate = 0.01
training_epochs = 1000
cost_history = np.empty(shape=[1], dtype=float)

# Train dataset to create a model
def train_model(x, weights, biases):
	# Hidden layer with RELU activation
	layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
	layer_1 = tf.nn.relu(layer_1)
	
	# Hidden layer with sigmoid activation
	layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b1'])
	layer_2 = tf.nn.relu(layer_2)
	# Hidden layer with sigmoid activation
	layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
	layer_3 = tf.nn.relu(layer_3)
    
    # Hidden layer with RELU activation
	layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
	layer_4 = tf.nn.relu(layer_4)
    
    # Output layer (must be transposed)
	output_layer = tf.transpose(tf.add(tf.matmul(layer_4, weights['out']), biases['out']))
	return output_layer

# Reduce loss and train a model repeatedly
def repeat_train_model(out, mse_history, x, y, sess, dir_path, x_train, y_train, x_test, y_test):
	# MSE function
	mse = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=y))
	#Optimizer
	optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(mse)
	# Calculate the cost and the accuracy for each epoch
	mse_history = []
	accuracy_history = []
	
	for epoch in range(training_epochs):
	    sess.run(optimizer, feed_dict={x:x_train, y:y_train})
	    cost = sess.run(mse, feed_dict={x:x_train, y:y_train})
	    mse_history = np.append(mse_history, cost)
	    correct_prediction = tf.equal(tf.argmax(out, 1), tf.argmax(y, 1))
	    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	    pred_y = sess.run(y, feed_dict={x:x_test})
	    mse = tf.reduce_mean(tf.square(pred_y - y_test))
	    mse_ = sess.run(mse)
	    mse_history.append(mse_)
	    accuracy = (sess.run(accuracy, feed_dict={x: x_train, y: y_train}))
	    accuracy_history.append(accuracy)
	    print('epoch:', epoch, '-', 'cost:', cost, "- MSE:", mse_, "- Train Accuracy: ", accuracy)
	
	saver = tf.train.Saver()
	save_path = saver.save(sess, dir_path + "/graph")
	print("Model saved in file: %s" % save_path)
	
	# Plot mse and accuracy graph
	plt.plot(mse_history, 'r')
	plt.show()
	plt.plot(accuracy_history)
	plt.show()
	return sess
	

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    