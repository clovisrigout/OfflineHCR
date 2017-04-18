import tensorflow as tf
import numpy as np
import cPickle as pickle # to load object to disk
import random


from data import Data
from cnnModel import cnnModel

def loadData():

    with open('../data/dataWithGen.pkl', 'rb') as input:
        dataO = pickle.load(input)
        return dataO

def loadModel():
	pass

def main():
	data = loadData()

	x = tf.placeholder(tf.float32, shape=[None, WIDTH,HEIGHT]) 
	y_ = tf.placeholder(tf.float32, shape=[None, 26])

	x_image = tf.reshape(x, [-1,WIDTH,HEIGHT,1]) #reshape to tensor

	W_conv1 = weight_variable([5, 5, 1, 32])
	b_conv1 = bias_variable([32])

	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
	h_pool1 = max_pool_2x2(h_conv1)

	W_conv2 = weight_variable([5, 5, 32, 128])
	b_conv2 = bias_variable([128])

	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
	h_pool2 = max_pool_2x2(h_conv2)

	W_fc1 = weight_variable([(WIDTH/2/2) * (HEIGHT/2/2) * 128, 4096])
	b_fc1 = bias_variable([4096])

	h_pool2_flat = tf.reshape(h_pool2, [-1, (WIDTH/2/2)*(HEIGHT/2/2)*128])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

	keep_prob = tf.placeholder(tf.float32)
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

	W_fc2 = weight_variable([4096, 26])
	b_fc2 = bias_variable([26])

	y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))
	train_step = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy)
	# train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
	correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	initializer = tf.global_variables_initializer()

	# Add ops to save and restore all the variables.
	saver = tf.train.Saver()

	with tf.Session() as sess:
		


if __name__ == "__main__":
	main()