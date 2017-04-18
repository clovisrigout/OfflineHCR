import tensorflow as tf
import numpy as np
import cPickle as pickle # to load object to disk
import sys
import math
from hmmlearn import hmm
from nltk.corpus import words as englishWords

from data import Data
from data import CUSTOM_GENERATION, DATA_OBJECT_PATHS

WIDTH = 28
HEIGHT = 28

ITERATIONS = 30000
NB_NEURONS_F = 120
NB_NEURONS_2 = 100
LEARNING_RATE = 1e-4

TRAIN_MODEL = True

alphabet = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"]

alphabetDict = {"a":0,"b":1,"c":2,"d":3,"e":4,"f":5,"g":6,"h":7,"i":8,"j":9,"k":10,"l":11,"m":12,"n":13,"o":14,"p":15,"q":16,"r":17,"s":18,"t":19,"u":20,"v":21,"w":22,"x":23,"y":24,"z":25}


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

def conv2d_Valid(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

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


def buildHmm():
	print "Building HMM"
	startProb = np.array([.11602, .04702, .03511, .02670, .02007, .03779, .01950, .07232, .06286, .00597, .00590, .02705, .04383, .02365, .06264, .02545, .00173, .01653, .07755, .16671, .01487, .00649, .06753, .00017, .01620, .00034])

	transitionProbs = []
	with open('../data/transition_prob.txt', 'rb') as file:
		index = 0
		for line in file:
			transitions = []
			print len(line.split())
			frequencies = [int(freq) for freq in line.split()]
			total = sum(frequencies)
			for freq in frequencies:
				transitions.append(float(freq)/total)
			transitionProbs.append(transitions)
			index = index+1
		print transitionProbs[0]
		print transitionProbs[25]

	transmat = np.array(transitionProbs)

	emissionProbs = [[0.7399497487437185, 0.007537688442211055, 0.007537688442211055, 0.016331658291457288, 0.00628140703517588, 0.0037688442211055275, 0.02135678391959799, 0.001256281407035176, 0.0, 0.0, 0.005025125628140704, 0.001256281407035176, 0.02135678391959799, 0.04648241206030151, 0.04396984924623116, 0.005025125628140704, 0.010050251256281407, 0.017587939698492462, 0.0037688442211055275, 0.00628140703517588, 0.01507537688442211, 0.0, 0.010050251256281407, 0.002512562814070352, 0.0, 0.007537688442211055], [0.0, 0.869198312236287, 0.004219409282700422, 0.016877637130801686, 0.012658227848101266, 0.008438818565400843, 0.004219409282700422, 0.02109704641350211, 0.004219409282700422, 0.0, 0.0, 0.004219409282700422, 0.004219409282700422, 0.0, 0.012658227848101266, 0.008438818565400843, 0.0, 0.0, 0.008438818565400843, 0.004219409282700422, 0.004219409282700422, 0.0, 0.0, 0.0, 0.0, 0.012658227848101266], [0.02030456852791878, 0.0, 0.8730964467005076, 0.005076142131979695, 0.04568527918781726, 0.005076142131979695, 0.0, 0.0025380710659898475, 0.005076142131979695, 0.0, 0.0, 0.0, 0.0, 0.0, 0.027918781725888325, 0.0, 0.0, 0.005076142131979695, 0.0025380710659898475, 0.0, 0.0025380710659898475, 0.0, 0.0, 0.0, 0.0025380710659898475, 0.0025380710659898475], [0.0392156862745098, 0.0784313725490196, 0.00392156862745098, 0.6196078431372549, 0.0, 0.027450980392156862, 0.047058823529411764, 0.027450980392156862, 0.00392156862745098, 0.03137254901960784, 0.00784313725490196, 0.023529411764705882, 0.00392156862745098, 0.00392156862745098, 0.023529411764705882, 0.00392156862745098, 0.00784313725490196, 0.0, 0.011764705882352941, 0.00392156862745098, 0.00784313725490196, 0.00784313725490196, 0.00784313725490196, 0.0, 0.0, 0.00784313725490196], [0.026172300981461286, 0.0054525627044711015, 0.026172300981461286, 0.0, 0.806979280261723, 0.013086150490730643, 0.013086150490730643, 0.0, 0.0, 0.0, 0.008724100327153763, 0.0010905125408942203, 0.0021810250817884407, 0.0, 0.008724100327153763, 0.025081788440567066, 0.0054525627044711015, 0.008724100327153763, 0.008724100327153763, 0.0010905125408942203, 0.004362050163576881, 0.0, 0.004362050163576881, 0.0, 0.0, 0.030534351145038167], [0.005025125628140704, 0.005025125628140704, 0.0, 0.010050251256281407, 0.020100502512562814, 0.8190954773869347, 0.005025125628140704, 0.005025125628140704, 0.0, 0.0, 0.0, 0.005025125628140704, 0.0, 0.005025125628140704, 0.0, 0.04522613065326633, 0.0, 0.01507537688442211, 0.005025125628140704, 0.04522613065326633, 0.0, 0.005025125628140704, 0.0, 0.0, 0.0, 0.005025125628140704], [0.020964360587002098, 0.012578616352201259, 0.0, 0.0041928721174004195, 0.0, 0.0041928721174004195, 0.7526205450733753, 0.0041928721174004195, 0.0, 0.012578616352201259, 0.0, 0.0, 0.0, 0.0041928721174004195, 0.025157232704402517, 0.010482180293501049, 0.039832285115303984, 0.006289308176100629, 0.05660377358490566, 0.0020964360587002098, 0.0, 0.0041928721174004195, 0.0, 0.0, 0.027253668763102725, 0.012578616352201259], [0.0, 0.08609271523178808, 0.0, 0.006622516556291391, 0.0, 0.006622516556291391, 0.0, 0.7284768211920529, 0.006622516556291391, 0.0, 0.033112582781456956, 0.046357615894039736, 0.006622516556291391, 0.026490066225165563, 0.0, 0.006622516556291391, 0.0, 0.006622516556291391, 0.0, 0.013245033112582781, 0.019867549668874173, 0.0, 0.006622516556291391, 0.0, 0.0, 0.0], [0.0, 0.008501594048884165, 0.008501594048884165, 0.0010626992561105207, 0.0, 0.04250797024442083, 0.009564293304994687, 0.005313496280552604, 0.7396386822529224, 0.024442082890541977, 0.0010626992561105207, 0.12646121147715197, 0.0, 0.0021253985122210413, 0.0010626992561105207, 0.005313496280552604, 0.0010626992561105207, 0.003188097768331562, 0.006376195536663124, 0.007438894792773645, 0.0, 0.003188097768331562, 0.0, 0.0010626992561105207, 0.0, 0.0021253985122210413], [0.0, 0.029411764705882353, 0.0, 0.029411764705882353, 0.0, 0.0, 0.058823529411764705, 0.0, 0.08823529411764706, 0.6176470588235294, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.029411764705882353, 0.0, 0.0, 0.029411764705882353, 0.058823529411764705, 0.0, 0.0, 0.0, 0.058823529411764705], [0.010471204188481676, 0.06806282722513089, 0.0, 0.020942408376963352, 0.010471204188481676, 0.03664921465968586, 0.005235602094240838, 0.05759162303664921, 0.010471204188481676, 0.0, 0.643979057591623, 0.005235602094240838, 0.015706806282722512, 0.010471204188481676, 0.005235602094240838, 0.010471204188481676, 0.0, 0.04712041884816754, 0.010471204188481676, 0.0, 0.005235602094240838, 0.0, 0.005235602094240838, 0.010471204188481676, 0.005235602094240838, 0.005235602094240838], [0.0, 0.009584664536741214, 0.03194888178913738, 0.0, 0.003194888178913738, 0.012779552715654952, 0.009584664536741214, 0.004792332268370607, 0.07188498402555911, 0.004792332268370607, 0.003194888178913738, 0.7891373801916933, 0.0, 0.0, 0.0, 0.019169329073482427, 0.003194888178913738, 0.0, 0.0, 0.01597444089456869, 0.0, 0.012779552715654952, 0.0, 0.001597444089456869, 0.006389776357827476, 0.0], [0.009836065573770493, 0.0, 0.0, 0.0, 0.0, 0.003278688524590164, 0.003278688524590164, 0.003278688524590164, 0.0, 0.0, 0.0, 0.0, 0.9081967213114754, 0.04262295081967213, 0.003278688524590164, 0.0, 0.0, 0.009836065573770493, 0.0, 0.0, 0.006557377049180328, 0.0, 0.009836065573770493, 0.0, 0.0, 0.0], [0.02920443101711984, 0.004028197381671702, 0.0, 0.004028197381671702, 0.0010070493454179255, 0.0030211480362537764, 0.0030211480362537764, 0.028197381671701913, 0.0010070493454179255, 0.0010070493454179255, 0.002014098690835851, 0.0010070493454179255, 0.03927492447129909, 0.798590130916415, 0.007049345417925478, 0.007049345417925478, 0.0010070493454179255, 0.012084592145015106, 0.0, 0.0030211480362537764, 0.03323262839879154, 0.005035246727089627, 0.014098690835850957, 0.0010070493454179255, 0.0010070493454179255, 0.0], [0.012482662968099861, 0.008321775312066574, 0.006934812760055479, 0.0013869625520110957, 0.004160887656033287, 0.006934812760055479, 0.011095700416088766, 0.0, 0.0013869625520110957, 0.005547850208044383, 0.0, 0.0013869625520110957, 0.0, 0.004160887656033287, 0.9112343966712899, 0.008321775312066574, 0.0, 0.0, 0.004160887656033287, 0.0, 0.0, 0.005547850208044383, 0.0013869625520110957, 0.0, 0.0027739251040221915, 0.0027739251040221915], [0.0, 0.0, 0.0, 0.0, 0.003745318352059925, 0.03745318352059925, 0.0, 0.0, 0.0, 0.0, 0.0, 0.018726591760299626, 0.0, 0.0, 0.0, 0.8913857677902621, 0.003745318352059925, 0.026217228464419477, 0.0, 0.003745318352059925, 0.0, 0.0, 0.0, 0.0, 0.00749063670411985, 0.00749063670411985], [0.08064516129032258, 0.0, 0.0, 0.0, 0.016129032258064516, 0.03225806451612903, 0.20967741935483872, 0.016129032258064516, 0.0, 0.0, 0.016129032258064516, 0.016129032258064516, 0.0, 0.0, 0.0, 0.12903225806451613, 0.3225806451612903, 0.0, 0.03225806451612903, 0.03225806451612903, 0.0, 0.0, 0.016129032258064516, 0.0, 0.03225806451612903, 0.04838709677419355], [0.006012024048096192, 0.004008016032064128, 0.014028056112224449, 0.0, 0.006012024048096192, 0.15430861723446893, 0.006012024048096192, 0.008016032064128256, 0.01002004008016032, 0.0, 0.008016032064128256, 0.006012024048096192, 0.018036072144288578, 0.01002004008016032, 0.0, 0.052104208416833664, 0.002004008016032064, 0.6132264529058116, 0.014028056112224449, 0.02404809619238477, 0.002004008016032064, 0.03006012024048096, 0.002004008016032064, 0.002004008016032064, 0.006012024048096192, 0.002004008016032064], [0.003472222222222222, 0.013888888888888888, 0.013888888888888888, 0.0, 0.0, 0.013888888888888888, 0.034722222222222224, 0.0, 0.003472222222222222, 0.003472222222222222, 0.0, 0.003472222222222222, 0.0, 0.003472222222222222, 0.006944444444444444, 0.0, 0.0, 0.0, 0.8888888888888888, 0.0, 0.0, 0.0, 0.0, 0.003472222222222222, 0.0, 0.006944444444444444], [0.0022624434389140274, 0.01583710407239819, 0.0022624434389140274, 0.0, 0.006787330316742082, 0.11538461538461539, 0.004524886877828055, 0.011312217194570135, 0.01583710407239819, 0.006787330316742082, 0.01809954751131222, 0.01583710407239819, 0.0, 0.0022624434389140274, 0.004524886877828055, 0.020361990950226245, 0.004524886877828055, 0.027149321266968326, 0.004524886877828055, 0.6923076923076923, 0.0022624434389140274, 0.0022624434389140274, 0.0, 0.004524886877828055, 0.00904977375565611, 0.011312217194570135], [0.053830227743271224, 0.018633540372670808, 0.002070393374741201, 0.004140786749482402, 0.0, 0.0, 0.002070393374741201, 0.018633540372670808, 0.002070393374741201, 0.002070393374741201, 0.006211180124223602, 0.0, 0.004140786749482402, 0.037267080745341616, 0.035196687370600416, 0.002070393374741201, 0.0, 0.0, 0.0, 0.0, 0.5714285714285714, 0.21739130434782608, 0.022774327122153208, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.007575757575757576, 0.007575757575757576, 0.0, 0.015151515151515152, 0.0, 0.0, 0.0, 0.0, 0.0, 0.030303030303030304, 0.0, 0.030303030303030304, 0.0, 0.015151515151515152, 0.0, 0.0, 0.05303030303030303, 0.8333333333333334, 0.007575757575757576, 0.0, 0.0, 0.0], [0.011363636363636364, 0.03409090909090909, 0.0, 0.011363636363636364, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.045454545454545456, 0.03409090909090909, 0.011363636363636364, 0.011363636363636364, 0.0, 0.0, 0.0, 0.0, 0.03409090909090909, 0.011363636363636364, 0.7954545454545454, 0.0, 0.0, 0.0], [0.03896103896103896, 0.012987012987012988, 0.0, 0.0, 0.0, 0.07792207792207792, 0.0, 0.0, 0.0, 0.0, 0.03896103896103896, 0.0, 0.025974025974025976, 0.0, 0.0, 0.025974025974025976, 0.0, 0.07792207792207792, 0.012987012987012988, 0.025974025974025976, 0.0, 0.012987012987012988, 0.0, 0.5844155844155844, 0.06493506493506493, 0.0], [0.0, 0.009852216748768473, 0.0, 0.014778325123152709, 0.0049261083743842365, 0.034482758620689655, 0.21674876847290642, 0.0, 0.0, 0.0, 0.0, 0.024630541871921183, 0.0, 0.0049261083743842365, 0.009852216748768473, 0.07389162561576355, 0.014778325123152709, 0.009852216748768473, 0.014778325123152709, 0.024630541871921183, 0.0049261083743842365, 0.014778325123152709, 0.0, 0.0049261083743842365, 0.5073891625615764, 0.009852216748768473], [0.02252252252252252, 0.009009009009009009, 0.0045045045045045045, 0.0, 0.036036036036036036, 0.0045045045045045045, 0.018018018018018018, 0.0045045045045045045, 0.0, 0.0045045045045045045, 0.013513513513513514, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.009009009009009009, 0.02702702702702703, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8468468468468469]]

	# model = hmm.GaussianHMM(algorithm='viterbi', startprob_prior=startProb, transmat_prior=transitionProbs, n_components=26, n_iter=100)
	model = hmm.MultinomialHMM(n_components=26)
	model.startprob_ = startProb
	model.transmat_ = transitionProbs
	model.emissionprob_ = emissionProbs

	return model

def makeOneHotVectors(labels):
	vectors = []
	for label in labels:
		vec = np.zeros(26)
		index = alphabetDict[label]
		vec[index] = 1
		vectors.append(vec)
	return vectors

def sigmoid(x):
	print(x)
	if x >= 0:
		z = math.exp(-x)
		return 1 / (1 + z)
	else:
		z = math.exp(x)
		return z / (1 + z)

def getPrefixes(w):
	prefixes = set("")
	print w
	if w == "":
		return None
	else:
		prefixes.add(w[0])
		othersPrefixes = getPrefixes(w[1:])
		if(othersPrefixes != None):
			for prefix in othersPrefixes:
				prefixes.add(w[0]+prefix)
		return prefixes

def getAccuracy(pred, word):
	total = len(word)
	count = 0
	for p,c in zip(pred, word):
		if(p == c):
			count += 1
	return float(count)/total

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
	correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	initializer = tf.global_variables_initializer()

	def getPredictions(X):
		return logits.eval(feed_dict={x: X, keep_prob: 1})

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

			# Save the variables to disk.
			if(CUSTOM_GENERATION):
				save_path = saver.save(sess, "./../models/model_2_with_gen_{}_{}_{}.ckpt".format(NB_NEURONS_2, NB_NEURONS_F, ITERATIONS))
			else:
				save_path = saver.save(sess, "./../models/model_2_{}_{}_{}.ckpt".format(NB_NEURONS_2, NB_NEURONS_F, ITERATIONS))
		else:
			if(CUSTOM_GENERATION):
				saver.restore(sess, "../models/model_2_with_gen_{}_{}_{}.ckpt".format(NB_NEURONS_2,NB_NEURONS_F, ITERATIONS))
			else:
				saver.restore(sess, "../models/model_2_{}_{}_{}.ckpt".format(NB_NEURONS_2,NB_NEURONS_F, ITERATIONS))
			handwritingAccuracy = accuracy.eval(feed_dict={x: data.handwritingX, y_: data.handwritingY, keep_prob: 1})
			print("handwritingAccuracy: %g"%handwritingAccuracy)
			predictions = getPredictions(data.handwritingX)
			# emissionProbs = getEmissionProbs_fast(data.testX, data.testY)
			# print("EMISSIONS:")
			# print(len(emissionProbs))
			# print(emissionProbs)
			# print(emissionProbs[0])

			print(len(predictions))
			print(len(data.handwritingY))
			# for prediction, actual in zip(predictions, data.handwritingY):
			# 	print(prediction)
			# 	print(actual)
			# 	print(np.matmul(np.transpose(prediction), actual))
			# 	print([sigmoid(x) for x in prediction])

			with open('../data/testText.txt', 'rb') as file:
				index = 0
				dictionary = set()
				for line in file:
					words = line.split()
					for word in words:
						dictionary.add(word)
			print("DICTONARY")
			print dictionary
			prefixes = set()
			for w in dictionary:
				prefixSet = getPrefixes(w)
				for prefix in prefixSet:
					prefixes.add(prefix)
			print("PREFIXES")
			print(prefixes)

			hmm = buildHmm()

			total = len(data.words)
			accuracies = []
			for word in data.words:
				imageD = [e[0] for e in word]
				labels = [e[1] for e in word]
				predictions = []
				for d in imageD:
					d = d.reshape((1,28,28))
					prediction = getPredictions(d)
					predictions.append(prediction)
				print(predictions[0])
				lengths = [len(predictions[0])]
				predictions = [np.argmax(pred) for pred in predictions]
				print(predictions)
				X = np.atleast_2d(predictions).T

				decoded = hmm.decode(X)
				print(decoded)
				sequence = decoded[1]
				prediction = []
				for c in sequence:
					prediction.append(alphabet[c])
				print prediction
				print(labels)
				accuracies.append(getAccuracy(prediction, labels))

			print "ACCURACIES"
			print(accuracies)
			wordAccuracy = np.mean(np.array(accuracies))
			print "wordAccuracy = ", wordAccuracy

			s = 0
			total =0
			for word, acc in zip(data.words, accuracies):
				l = len(word)
				s += l*acc
				total += l

			meanAccuracy = float(s)/total
			print "meanAccuracy = ", meanAccuracy
			writer = tf.summary.FileWriter('../graphs', sess.graph)
			writer.close()



if __name__ == "__main__":
	main()