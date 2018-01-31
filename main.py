# -*- coding: utf-8 -*-
# Created on Fri Jan 12 18:37:45 2018
# @author: acer
# =====================================

"""Main module."""

import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import data_preprocessing as dp
import model_training as mtn
import model_testing as mtt

# =======================Step 1: Data Pre-Processing===========================
# Get current directory path
dir_path = os.getcwd()
dataset_path = dir_path + "/cpi_dataset.csv" 

# Read training dataset
data = dp.read_dataset(dataset_path)
# Dimension of data
n = data.shape[0]
p = data.shape[1]

# Training and Testing dataset
train_start = 0;
train_end = int(np.floor(n*0.8))
test_start = train_end
test_end = n
data_train = data[np.arange(train_start, train_end), :]
data_test = data[np.arange(test_start, test_end), :]

# Scale data
scaler = MinMaxScaler()
scaler.fit(data_train)
data_train = scaler.transform(data_train)
data_test = scaler.transform(data_test)

# Build X and Y
x_train = data_train[:, 1:]
y_train = data_train[:, 0]
x_test = data_test[:, 1:]
y_test = data_test[:, 0]
#
# =======================Step 2: Training Model================================
# Model architecture parameters
n_hidden_1 = 60
n_hidden_2 = 60
n_hidden_3 = 60
n_hidden_4 = 60
n_class = 1

# Placeholder
x = tf.placeholder(tf.float32, [1, n])
y = tf.placeholder(tf.float32, [None, n_class])

# Calculate the cost and the accuracy for each epoch
mse_history = []
accuracy_history = []

# Define the weights and the biases for each layer
weights = {
        'h1': tf.Variable(tf.random_normal([n, n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
        'h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4])),
        'out': tf.Variable(tf.random_normal([n_hidden_4, n_class]))
        }
biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'b3': tf.Variable(tf.random_normal([n_hidden_3])),
        'b4': tf.Variable(tf.random_normal([n_hidden_4])),
        'out': tf.Variable(tf.random_normal([n_class]))
        }
sess = tf.Session()
output = mtn.train_model(x, weights, biases)
sess = mtn.repeat_train_model(output, mse_history, x, y, sess, dir_path, x_train, y_train, x_test, y_test)
# 
# =======================Step 3: Testing Model=================================
#mtt.test_model(sess, output, x, y, x_test, y_test)
#

