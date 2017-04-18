import tensorflow as tf
import numpy as np
import cPickle as pickle # to load object to disk
import sys 
from data import Data
from data import CUSTOM_GENERATION, DATA_OBJECT_PATHS

WIDTH = 28
HEIGHT = 28

TRAIN_MODEL = True

def loadData():

	path = DATA_OBJECT_PATHS[CUSTOM_GENERATION]
	print("loading data at path {}".format(path))
	with open(path, 'rb') as input:
		dataO = pickle.load(input)
		return dataO

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

# The convolution uses a stride of one and are zero padded so that the output is the same size as the input.
def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

#Our pooling is plain old max pooling over 2x2 blocks
#output divides size by two
def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def max_pool_4x4(x):
	return tf.nn.max_pool(x, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')

class cnnModel:
	def __init__(self, W, b):
		self.W = W
		self.b = b

def main():

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

		data = loadData()

		if(TRAIN_MODEL):

			sess.run(initializer)

			for i in range(3000):
				print("Current Batch Index: {}".format(data.getBatchIndex()))
				batch = data.nextTrainBatch(50)
				if i%100 == 0:
					train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
					print("step %d, training accuracy %g"%(i, train_accuracy))

				train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.85})

			testAccuracy = accuracy.eval(feed_dict={x: data.testX, y_: data.testY, keep_prob: 1.0})

			print("test accuracy %g"%testAccuracy)

			alphabetAccuracy = accuracy.eval(feed_dict={x: data.alphabetX, y_: data.alphabetY, keep_prob: 1.0})

			print("alphabetAccuracy %g"%alphabetAccuracy)

			handwritingAccuracy = accuracy.eval(feed_dict={x: data.handwritingX, y_: data.handwritingY, keep_prob: 1.0})
			print("handwritingAccuracy: %g"%handwritingAccuracy)

			# Save the variables to disk.
			save_path = saver.save(sess, "./../models/model_with_gen_4096_5000.ckpt")

		else:
			saver.restore(sess, "../models/model_with_gen_4096_5000.ckpt")
			handwritingAccuracy = accuracy.eval(feed_dict={x: data.handwritingX, y_: data.handwritingY, keep_prob: 1.0})
			print("handwritingAccuracy: %g"%handwritingAccuracy)


if __name__ == "__main__":
	main()