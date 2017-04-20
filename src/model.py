import tensorflow as tf
import numpy as np
import cPickle as pickle # to load object to disk
import sys
import math
from hmmlearn import hmm
from nltk.corpus import words as englishWords
from PIL import Image
import os

from data import Data, CUSTOM_GENERATION, DATA_OBJECT_PATHS

WIDTH = 28
HEIGHT = 28

ITERATIONS = 20000
NB_NEURONS_F = 120
NB_NEURONS_2 = 100
LEARNING_RATE = 1e-4

TRAIN_MODEL = True
WITH_LM = True
WITH_DICTIONARY = True

alphabet = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"]

alphabetDict = {"a":0,"b":1,"c":2,"d":3,"e":4,"f":5,"g":6,"h":7,"i":8,"j":9,"k":10,"l":11,"m":12,"n":13,"o":14,"p":15,"q":16,"r":17,"s":18,"t":19,"u":20,"v":21,"w":22,"x":23,"y":24,"z":25}


class languageModel:

	def __init__(self, startprob, prob, transitionmat):
		self.startprob = startprob
		self.prob = prob
		self.transitionmat = transitionmat


def main():

	# see https://www.tensorflow.org/tutorials/layers#input_layer
	x = tf.placeholder(tf.float32, shape=[None, WIDTH,HEIGHT]) 
	y_ = tf.placeholder(tf.float32, shape=[None, 26])

	input_layer = tf.reshape(x, [-1,WIDTH,HEIGHT,1]) #reshape to tensor

	conv1 = tf.layers.conv2d(
	    inputs=input_layer,
	    filters=6,
	    kernel_size=[5, 5],
	    padding="same",
	    activation=tf.nn.relu
	)

	pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

	conv2 = tf.layers.conv2d(
	    inputs=pool1,
	    filters=16,
	    kernel_size=[5, 5],
	    padding="valid",
	    activation=tf.nn.relu
	)

	pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

	pool2_flat = tf.reshape(pool2, [-1, 5 * 5 * 16])

	dense1 = tf.layers.dense(inputs=pool2_flat, units=NB_NEURONS_2, activation=tf.nn.relu)

	dense2 = tf.layers.dense(inputs=dense1, units=NB_NEURONS_F, activation=tf.nn.relu)

	keep_prob = tf.placeholder(tf.float32)
	dropout = tf.layers.dropout(
	    inputs=dense2, rate=keep_prob
	)

	logits = tf.layers.dense(inputs=dropout, units=26)

	predictions = {
	    "classes": tf.argmax(
	        input=logits, axis=1),
	    "probabilities": tf.nn.softmax(
	        logits, name="softmax_tensor")
	}

	loss = tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=y_)
	train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
	# train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
	correct_prediction = tf.equal(predictions["classes"], tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	initializer = tf.global_variables_initializer()

	# Add ops to save and restore all the variables.
	saver = tf.train.Saver()

	with tf.Session() as sess:

		data = loadData()

		if(TRAIN_MODEL):
			sess.run(initializer)

			for i in range(ITERATIONS):
				print("Current Batch Index: {}".format(data.getBatchIndex()))
				batch = data.nextTrainBatch(50)
				if i%100 == 0:
					train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1})
					print("step %d, training accuracy %g"%(i, train_accuracy))

				train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

			train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_:batch[1], keep_prob: 1})
			print("train_accuracy %g"%train_accuracy)

			testAccuracy = accuracy.eval(feed_dict={x: data.testX, y_: data.testY, keep_prob: 1.0})
			print("test accuracy %g"%testAccuracy)

			alphabetAccuracy = accuracy.eval(feed_dict={x: data.alphabetX, y_: data.alphabetY, keep_prob: 1.0})
			print("alphabetAccuracy %g"%alphabetAccuracy)

			handwritingAccuracy = accuracy.eval(feed_dict={x: data.handwritingX, y_: data.handwritingY, keep_prob: 1.0})
			print("handwritingAccuracy: %g"%handwritingAccuracy)

			# Save the model to disk.
			if(CUSTOM_GENERATION):
				save_path = saver.save(sess, "./../models/model_with_gen_{}_{}_{}.ckpt".format(NB_NEURONS_2, NB_NEURONS_F, ITERATIONS))
			else:
				save_path = saver.save(sess, "./../models/model_{}_{}_{}.ckpt".format(NB_NEURONS_2, NB_NEURONS_F, ITERATIONS))

		else:
			if(CUSTOM_GENERATION):
				try:
					saver.restore(sess, "../models/model_with_gen_{}_{}_{}.ckpt".format(NB_NEURONS_2,NB_NEURONS_F, ITERATIONS))
				except:
					print "Didn't find model with the following parameters. You may need to train it first."
					exit(1)
			else:
				try:
					saver.restore(sess, "../models/model_{}_{}_{}.ckpt".format(NB_NEURONS_2,NB_NEURONS_F, ITERATIONS))
				except:
					print "Didn't find model with the following parameters. You may need to train it first."
					exit(1)

			handwritingAccuracy = accuracy.eval(feed_dict={x: data.handwritingX, y_: data.handwritingY, keep_prob: 1})
			print("handwritingAccuracy: %g"%handwritingAccuracy)

			predictions = []
			for word in data.words:
				imageD = [e[0] for e in word]
				wordLabel = "".join([e[1] for e in word])
				wordPredictions = []
				for d in imageD:
					d = d.reshape((1,28,28))
					prediction = getPredictions(d)
					wordPredictions.append(prediction[0])
				predictions.append(wordPredictions)

			if(WITH_LM):
				lm = buildEnglishLM()

				if(WITH_DICTIONARY):
					dictionary = getDictionary()
					# maximize likelihood over dictionary:
					wordPreds = []
					wordTrues = []
					for word, preds in zip(data.words, predictions):
						labels = [e[1] for e in word]
						maxLikelihood = -float('inf')
						lengthKWords = [w for w in list(dictionary) if len(w) == len(word)]
						wordPred = lengthKWords[0]
						for w in lengthKWords:
							likelihood = getLikelihood(w, lm, preds)
							if(likelihood > maxLikelihood):
								maxLikelihood = likelihood
								wordPred = w
						wordTrue = "".join(labels)
						wordPreds.append(wordPred)
						wordTrues.append(wordTrue)
						print("Prediction : ", wordPred)
						print("Actual: ", wordTrue)

					acc = getWordAccuracy(wordPreds, wordTrues)

					acc = getHandwritingAccuracy(wordPreds, wordTrues)

				else:

					# maximize likelihood over all english words:
					wordPreds = []
					wordTrues = []
					for word, preds in zip(data.words, predictions):
						labels = [e[1] for e in word]
						maxLikelihood = -float('inf')
						lengthKWords = getAllLengthKWords(len(word))
						wordPred = lengthKWords[0]
						for w in lengthKWords:
							likelihood = getLikelihood(w, lm, preds)
							if(likelihood > maxLikelihood):
								maxLikelihood = likelihood
								wordPred = w
						wordTrue = "".join(labels)
						wordPreds.append(wordPred)
						wordTrues.append(wordTrue)
						print("Prediction : ", wordPred)
						print("Actual: ", wordTrue)

					acc = getWordAccuracy(wordPreds, wordTrues)

					acc = getHandwritingAccuracy(wordPreds, wordTrues)

			else:
				# language model with probability of 1 everywhere... as if no model at all
				lm = languageModel([1 for i in range(26)], [1 for i in range(26)], [[1 for i in range(26)] for x in range(26)] )  
				wordPreds = []
				wordTrues = []
				for word, preds in zip(data.words, predictions):
					labels = [e[1] for e in word]
					maxLikelihood = -float('inf')
					lengthKWords = getAllLengthKWords(len(word))
					wordPred = lengthKWords[0]
					for w in lengthKWords:
						likelihood = getLikelihood(w, lm, preds)
						if(likelihood > maxLikelihood):
							maxLikelihood = likelihood
							wordPred = w
					wordTrue = "".join(labels)
					wordPreds.append(wordPred)
					wordTrues.append(wordTrue)
					print("Prediction : ", wordPred)
					print("Actual: ", wordTrue)

				acc = getWordAccuracy(wordPreds, wordTrues)
				acc = getHandwritingAccuracy(wordPreds, wordTrues)


######################################################################################################
######################################################################################################
########################################## HELPER METHODS ############################################
######################################################################################################
######################################################################################################


def loadData():

	path = DATA_OBJECT_PATHS[CUSTOM_GENERATION]
	print("loading data at path {}".format(path))
	with open(path, 'rb') as input:
		dataO = pickle.load(input)
		return dataO


def buildEnglishLM():
	startprob = np.array([.11602, .04702, .03511, .02670, .02007, .03779, .01950, .07232, .06286, .00597, .00590, .02705, .04383, .02365, .06264, .02545, .00173, .01653, .07755, .16671, .01487, .00649, .06753, .00017, .01620, .00034])

	# from https://www.math.cornell.edu/~mec/2003-2004/cryptography/subs/frequencies.html
	prob = np.array([.0812, 0.0149, 0.0271, 0.0432, 0.1202, 0.0230, 0.0204, 0.0592, 0.0731, 0.0010, 0.0069, 0.0398, 0.0261, 0.0695, 0.0768, 0.0182, 0.0011, 0.0602, 0.0628, 0.0910, 0.0288, 0.0111, 0.0209, 0.0017, 0.0211, 0.0007])

	transitionmat = []
	with open('../data/transition_prob.txt', 'rb') as file:
		index = 0
		for line in file:
			transitions = []
			frequencies = [int(freq) for freq in line.split()]
			total = sum(frequencies)
			for freq in frequencies:
				transitions.append(float(freq)/total)
			transitionmat.append(transitions)
			index = index+1

	return languageModel(startprob, prob, transitionmat)

def makeOneHotVectors(labels):
	vectors = []
	for label in labels:
		vec = np.zeros(26)
		index = alphabetDict[label]
		vec[index] = 1
		vectors.append(vec)
	return vectors


def getPredictions(X, kind="probabilities"):
	predictions = {
    "classes": tf.argmax(
        input=logits, axis=1),
    "probabilities": tf.nn.softmax(
        logits, name="softmax_tensor")
	}
	return predictions[kind].eval(feed_dict={x: X, keep_prob: 1})

def predict(X):
	X = [X]
	predictions = getPredictions(X)
	argmax = np.argmax(predictions)
	prediction = makeOneHotVectors(alphabet[argmax])
	return prediction

def getEmissionProbs(X, Y):
	data = zip(X, Y)
	states = makeOneHotVectors(alphabet)
	emissionProbs = []
	for trueState in states:
		trueData = [(o,s) for (o,s) in data if(np.dot(s,trueState) == 1)]
		total = len(trueData)
		emissions = []
		for predState in states:
			nbPred = sum(1 for (o,_) in trueData if(np.dot(predict(o),predState) == 1 ))
			emissions.append(float(nbPred)/total)
		emissionProbs.append(emissions)
	return emissionProbs

def getEmissionProbs_fast(X, Y):
	data = zip(X, Y)
	emissionProbs = []
	for i, true_label in enumerate(alphabet):
		trueData = [(o,s) for (o,s) in data if(np.argmax(s) == i)]
		total = len(trueData)
		emissions = []
		for j, pred_label in enumerate(alphabet):
			nbPred = sum(1 for (o,_) in trueData if(np.argmax(predict(o)) == j))
			emissions.append(float(nbPred)/total)
		emissionProbs.append(emissions)
	return emissionProbs

def getEmissionProbs2(X, Y):
	data = zip(X, Y)
	states = makeOneHotVectors(alphabet)
	emissionProbs = []
	for s in states:
		z_s = sum(getCustomProb(o, s) for o in X)
		probs = []
		for i in X:
			prob = getCustomProb(o,s) / z_s
			prob.append(prob)
		emissionProbs.append(probs)
	return emissionProbs

def getCustomProb(O, S):
	# sort values keeping original indices
	o = sorted(enumerate(O), key=lambda x: x[1], reverse=True)
	# make oneHotVectors using the original indices
	o = [makeOneHotVectors(alphabet[i]) for (i, _) in o]
	# multiply by 1/(i+1) to reduce possible score
	o = [o_i/(i+1) for i, o_i in enumerate(o)]
	prob = sum(np.dot(S, o_i) for o_i in o)
	return prob


def getLikelihood(word, lm, preds):
	initialIndex = alphabetDict[word[0].lower()]
	likelihood = lm.startprob[initialIndex]*preds[0][initialIndex]
	previousIndex = initialIndex
	for w_i, p in zip(word[1:], preds[1:]):
		index = alphabetDict[w_i.lower()]
		likelihood *=  lm.prob[index]*p[index]*lm.transitionmat[previousIndex][index]
		previousIndex = index
	return likelihood


def getAllLengthKWords(k):
	lengthKWords = [word.split("-")[0].lower() for word in englishWords.words() if len(word.split("-")[0]) == k]
	if k == 6:
		lengthKWords.append("clovis")
	return lengthKWords

def getDictionary():
	dictionary = set()
	with open('../data/testText.txt', 'rb') as file:
		index = 0
		for line in file:
			words = line.split()
			for word in words:
				dictionary.add(word)
	return dictionary


def getWordAccuracy(wordPreds, wordTrues):
	total = len(wordPreds)
	correct = 0
	for pred, t in zip(wordPreds, wordTrues):
		if(pred == t):
			correct += 1
	acc = float(correct)/total
	print "WORD ACCURACY = {}".format(acc)
	return acc

def getHandwritingAccuracy(wordPreds, wordTrues):
	total = sum(len(wordPred) for wordPred in wordPreds)
	correct = 0
	for wordPred, wordTrue in zip(wordPreds, wordTrues):
		for w_1i, w_2i in zip(wordPred, wordTrue):
			if(w_1i == w_2i):
				correct += 1
	acc = float(correct)/total
	print "HW ACCURACY = {}".format(acc)
	return acc

if __name__ == "__main__":
	main()