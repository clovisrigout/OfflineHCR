import tensorflow as tf
import numpy as np
import cPickle as pickle # to load object to disk
from data import Data

def loadData():

    with open('../data/data.pkl', 'rb') as input:
        dataO = pickle.load(input)
        return dataO

def getLossFunction(y, y_):
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
	return cross_entropy

def getModel(x, W, b):
	y = tf.matmul(x,W) + b
	return y

def trainModel(data,x,y_,loss):
	train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
	for _ in range(1000):
		print("Current Batch Index: {}".format(data.getBatchIndex()))
		batch = data.nextTrainBatch(100)
		xBatch = [m.flatten() for m in batch[0]]
		# print("HELLO")
		# print(batch[1])
		train_step.run(feed_dict={x: xBatch, y_: batch[1]})

def getAccuracy(y, y_):
	correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	return accuracy


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
def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def main():

	#input images : 784 is the dimensionality of a single flattened 24 by 24 pixel MNIST image
	x = tf.placeholder(tf.float32, shape=[None, 1024]) 
	y_ = tf.placeholder(tf.float32, shape=[None, 26]) #target output classes

	# W is a 784x10 matrix (because we have 784 input features and 26 outputs (alphabet)
	W = tf.Variable(tf.zeros([1024,26])) 
	b = tf.Variable(tf.zeros([26]))

	data = loadData()

	y = getModel(x,W,b)

	loss = getLossFunction(y,y_);

	initializer = tf.initialize_all_variables()

	with tf.Session() as sess:

		sess.run(initializer)

		trainModel(data, x, y_, loss)

		accuracy = getAccuracy(y, y_)

		writer = tf.summary.FileWriter('../graphs', sess.graph)

		xData = [m.flatten() for m in data.testX]

		customX = [m.flatten() for m in data.additionalX]
		customY = [m.flatten() for m in data.additionalY]

		print(accuracy.eval(feed_dict={x: customX, y_:customY}))
		print(accuracy.eval(feed_dict={x: xData, y_: data.testY}))

		writer.close()

if __name__ == "__main__":
	main()