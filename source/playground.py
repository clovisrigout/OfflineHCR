import tensorflow as tf
import numpy as np
import cPickle as pickle # to load object to disk
import random


from data import Data
from cnnModel import cnnModel

def loadData():

    with open('../data/data.pkl', 'rb') as input:
        dataO = pickle.load(input)
        return dataO

def loadModel():

	with open('../models/model.pkl', 'rb') as input:
		modelO = pickle.load(input)
		return modelO

def main():
	data = loadData()

	model = loadModel()

	weights = np.array(model.W)
	bias = np.array(model.b)

	for (image, label) in zip(data.additionalX, data.additionalY):
		print("LABEL {}".format(label))
		pred = np.add(bias,np.matmul(image, weights))
		print(pred)


if __name__ == "__main__":
	main()