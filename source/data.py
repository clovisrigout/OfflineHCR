# Copyright 2011 Hugo Larochelle. All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without modification, are
# permitted provided that the following conditions are met:
# 
#    1. Redistributions of source code must retain the above copyright notice, this list of
#       conditions and the following disclaimer.
# 
#    2. Redistributions in binary form must reproduce the above copyright notice, this list
#       of conditions and the following disclaimer in the documentation and/or other materials
#       provided with the distribution.
# 
# THIS SOFTWARE IS PROVIDED BY Hugo Larochelle ``AS IS'' AND ANY EXPRESS OR IMPLIED
# WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
# FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL Hugo Larochelle OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# 
# The views and conclusions contained in the software and documentation are those of the
# authors and should not be interpreted as representing official policies, either expressed
# or implied, of Hugo Larochelle.

"""
Module ``datasets.ocr_letters`` gives access to the OCR letters dataset.

The OCR letters dataset was first obtained here: http://ai.stanford.edu/~btaskar/ocr/letter.data.gz.

| **Reference:** 
| Tractable Multivariate Binary Density Estimation and the Restricted Boltzmann Forest
| Larochelle, Bengio and Turian
| http://www.cs.toronto.edu/~larocheh/publications/NECO-10-09-1100R2-PDF.pdf

"""

import mlpython.misc.io as mlio
import numpy as np
import os
from gzip import GzipFile as gfile

def load(dir_path,load_to_memory=False,load_as_images=False):
    """
    Loads the OCR letters dataset.

    The data is given by a dictionary mapping from strings
    ``'train'``, ``'valid'`` and ``'test'`` to the associated pair of data and metadata.
    
    **Defined metadata:**
    
    * ``'input_size'``
    * ``'targets'``
    * ``'length'``

    """
    
    input_size=128
    targets = set(range(26))
    dir_path = os.path.expanduser(dir_path)

    if load_as_images:
        def load_line(line):
            tokens = line.split()
            return (np.array([int(i)*255 for i in tokens[:-1]],dtype='uint8').reshape((16,8)),tokens[-1])
    else:
        def load_line(line):
            tokens = line.split()
            return (np.array([float(i) for i in tokens[:-1]]),int(tokens[-1]))

    train_file,valid_file,test_file = [os.path.join(dir_path, 'ocr_letters_' + ds + '.txt') for ds in ['train','valid','test']]
    # Get data
    train,valid,test = [mlio.load_from_file(f,load_line) for f in [train_file,valid_file,test_file]]

    lengths = [32152,10000,10000]

    if load_to_memory:
        if load_as_images:
            train,valid,test = [mlio.MemoryDataset(d,[((16,8)),(1,)],[np.uint8,int],l) for d,l in zip([train,valid,test],lengths)]
        else:
            train,valid,test = [mlio.MemoryDataset(d,[(input_size,),(1,)],[np.float64,int],l) for d,l in zip([train,valid,test],lengths)]

    # Get metadata
    train_meta,valid_meta,test_meta = [{'input_size':input_size,
                              'length':l,'targets':targets} for l in lengths]

    return {'train':(train,train_meta),'valid':(valid,valid_meta),'test':(test,test_meta)}

def obtain(dir_path):
    """
    Downloads the dataset to ``dir_path``.
    """

    dir_path = os.path.expanduser(dir_path)
    print 'Downloading the dataset'
    import urllib
    #urllib.urlretrieve('http://ai.stanford.edu/~btaskar/ocr/letter.data.gz',os.path.join(dir_path,'letter.data.gz'))
    urllib.urlretrieve('http://info.usherbrooke.ca/hlarochelle/public/letter.data.gz',os.path.join(dir_path,'letter.data.gz'))

    print 'Splitting dataset into training/validation/test sets'
    file = gfile(os.path.join(dir_path,'letter.data.gz'))
    train_file,valid_file,test_file = [open(os.path.join(dir_path, 'ocr_letters_' + ds + '.txt'),'w') for ds in ['train','valid','test']]
    letters = 'abcdefghijklmnopqrstuvwxyz'
    all_data = []
    # Putting all data in memory
    for line in file:
        tokens = line.strip('\n').strip('\t').split('\t')
        s = ''        
        for t in range(6,len(tokens)):
            s = s + tokens[t] + ' '
        target = letters.find(tokens[1])
        if target < 0:
            print 'Target ' + tokens[1] + ' not found!'
        s = s + str(target) + '\n'
        all_data += [s]

    # Shuffle data
    import random
    random.seed(12345)
    perm = range(len(all_data))
    random.shuffle(perm)
    line_id = 0
    train_valid_split = 32152
    valid_test_split = 42152
    for i in perm:
        s = all_data[i]
        if line_id < train_valid_split:
            train_file.write(s)
        elif line_id < valid_test_split:
            valid_file.write(s)
        else:
            test_file.write(s)
        line_id += 1
    train_file.close()
    valid_file.close()
    test_file.close()
    print 'Done'


###############################################################################################
####################################### CUSTOM CODE ###########################################
###############################################################################################
from PIL import Image, ImageOps
import glob
import cPickle as pickle #to save object to disk
import random
np.set_printoptions(threshold=np.inf)


alphabet = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"]

alphabetDict = {"a":0,"b":1,"c":2,"d":3,"e":4,"f":5,"g":6,"h":7,"i":8,"j":9,"k":10,"l":11,"m":12,"n":13,"o":14,"p":15,"q":16,"r":17,"s":18,"t":19,"u":20,"v":21,"w":22,"x":23,"y":24,"z":25}

ADDITIONAL_DATA_PATH = "../data/paint_data"
ALPHABET_DATA_PATH = "../data/formatted_data"

IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28

RELOAD_DATA = True
CENTER_IMAGES = False


def cropImage(image):
    imD = image.load()
    (X, Y) = image.size
    m = np.zeros((X, Y))

    topIndex = Y
    leftIndex = X
    rightIndex = 0
    bottomIndex = 0
    for x in range(X):
        for y in range(Y):
            if(imD[(x, y)] != (255,255,255,255)):
                if(topIndex) > y:
                    topIndex = y
                if(leftIndex) > x:
                    leftIndex = x
                if(rightIndex) < x:
                    rightIndex = x
                if(bottomIndex) < y:
                    bottomIndex = y
    return image.crop((leftIndex, topIndex, rightIndex, bottomIndex))


def findCenterOfMass(image):
    imD = image.load()
    (X, Y) = image.size
    m = np.zeros((X, Y))

    for x in range(X):
        for y in range(Y):
            m[x, y] = imD[(x, y)] != (255, 255, 255, 255)
    m = m / np.sum(m)

    # marginal distributions
    dx = np.sum(m, axis=0)
    dy = np.sum(m, axis=1)

    # expected values
    cx = np.sum(dx * np.arange(X))
    cy = np.sum(dy * np.arange(Y))

    return (cx,cy)

def getImageAndLabelFromMLPythonExample(example, save_image=False, center_images=False):
    # example[0] = image array
    # example[1] = image label (int)
    # print(example)
    # print int(example[1])
    # print(alphabet[int(example[1])])
    im = Image.fromarray(np.array(example[0]))
    im = invertColor(im)
    im = cropImage(im)
    im = im.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.ANTIALIAS)
    if(center_images):
        im = recenterImage(im)
    if(save_image):
        im.save("./Stanford_OCR/"+alphabet[int(example[1])]+".png")
    return im, alphabet[int(example[1])]

def imageToData(image):
    image = setWhiteBackground(image)                       # set white background
    image = cropImage(image)                                # crop
    image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.ANTIALIAS)  # resize
    m = makeBinaryMatrixFromImage(image)                    # make binary image: 1=black

    return m, image


def invertColor(image):
    return ImageOps.invert(image)


def loadAdditionalData(dir_path, save_images=False, center_images=False):
    dir_path = os.path.expanduser(dir_path)

    data = []
    labels = []

    for filePath in glob.glob(dir_path +"/*.png"):
        fileName = filePath.split(".png")[0].split("/")[-1]
        image = Image.open(filePath)
        imageData, im = imageToData(image)
        data.append(imageData)

        labels.append(fileName)
        if(save_images):
            savePath = "./formatted_data/"+fileName+".png"
            if(center_images):
                im = recenterImage(im)
                im = im.convert("RGB").save(savePath)
            else:
                im = im.convert("RGB").save(savePath)
    return data,labels

def loadFormattedCustomAlphabet(dir_path):
    dir_path = os.path.expanduser(dir_path)

    data = []
    labels = []

    for filePath in glob.glob(dir_path +"/*.png"):
        fileName = filePath.split(".png")[0].split("/")[-1]
        image = Image.open(filePath)
        imageData = makeBinaryMatrixFromImage(image)               # make binary image: 1=black
        data.append(imageData)

        labels.append(fileName)
    return data,labels


def makeBinaryMatrixFromImage(image):
    bwImage = image.convert('1')                            # convert to black and white
    A = np.array(bwImage)                                   
    bwA = np.empty((A.shape[0], A.shape[1]), None)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i][j] == True:
                bwA[i][j] = 0
            else:
                bwA[i][j] = 1
    return bwA

def makeOneHotVectors(labels):
    vectors = []
    for label in labels:
        vec = np.zeros(26)
        index = alphabetDict[label]
        vec[index] = 1
        vectors.append(vec)
    return vectors

def recenterImage(image):
    (cx, cy) = findCenterOfMass(image)
    im = shiftImageCenterToPoint(image, (int(cx), int(cy)), (int(IMAGE_WIDTH/2),int(IMAGE_HEIGHT/2)))
    return im

def setWhiteBackground(image):
    A = np.array(image)
    for x in xrange(A.shape[0]):
        for y in xrange(A.shape[1]):
            if A[x][y][3] < 10 :
                A[x][y] = [255,255,255,255]
    image = Image.fromarray(A)
    return image


def shiftImageCenterToPoint(image, (cx,cy), (px,py)):
    imD = image.load()
    (X, Y) = image.size
    m = np.zeros((X, Y), dtype='4uint8')
    #initialize white background
    for x in range(X):
        for y in range(Y):
            m[(x,y)] = (255,255,255,255)

    for x in range(X): # x is column
        for y in range(Y): # y is row
            if(imD[(y, x)] != (255,255,255,255)):
                diffx = cx-x
                diffy = cy-y
                shiftx = px-diffx
                shifty = py-diffy
                if((0 <= shiftx < X) and (0<= shifty < Y)):
                    m[(shiftx, shifty)] = imD[(y, x)]

    centeredImage = Image.fromarray(np.array(m))
    return centeredImage


class Data:
    def __init__(self,trainX, trainY, testX, testY):
        self.trainX = trainX
        self.trainY = trainY
        self.train = [trainX, trainY]
        self.testX = testX
        self.testY = testY
        self.test = [testX, testY]
        self.additionalX = []
        self.additionalY = []
        self.batchIndex = 0

    def addAdditionData(self, additionalX, additionalY):
        self.additionalX = additionalX
        self.additionalY = additionalY

    def nextTrainBatch(self, size):
        if(self.batchIndex+size >= len(self.trainX)):
            batchX = self.trainX[self.batchIndex:]
            batchY = self.trainY[self.batchIndex:]
            self.batchIndex=0
        else:
            batchX = self.trainX[self.batchIndex:self.batchIndex+size]
            batchY = self.trainY[self.batchIndex:self.batchIndex+size]
            self.batchIndex += size
        return [batchX,batchY]

    def getBatchIndex(self):
        return self.batchIndex

    def getTrainDataFromLabel(self, label):
        classIndices = [index for index, y in enumerate(self.trainY) if (y[label] == 1)]
        images = [self.trainX[i] for i in classIndices]
        oneHot = np.zeros(26)
        oneHot[label] = 1
        return images, oneHot

    def balanceDataClasses(self):
        counts = np.zeros(26)
        for y in self.trainY:
            counts = np.add(counts, y)

        needed = np.zeros(26)
        needed.fill(np.amax(counts))
        needed = np.subtract(needed, counts)

        newX = []
        newY = []
        for i, remaining in enumerate(needed):
            images, label = self.getTrainDataFromLabel(i)
            while(remaining > 0):
                new = images[random.randint(0, len(images)-1)]
                newX.append(new)
                newY.append(label)
                remaining += -1

        newData = zip(newX, newY)

        random.shuffle(newData)

        newX = [d[0] for d in newData]
        newY = [d[1] for d in newData]

        self.trainX.extend(newX)
        self.trainY.extend(newY)

    def shuffleTrainData(self):
        zipData = zip(self.trainX, self.trainY)
        random.shuffle(zipData)
        shuffledX = [d[0] for d in zipData]
        shuffledY = [d[1] for d in zipData]

        self.trainX = shuffledX
        self.trainY = shuffledY

    def generateTransformedData(data, counts):

        for d, count in zip(data,counts):
            



def main():
    if(RELOAD_DATA):
        # obtain("./Stanford_OCR") #download dataset
        data = load("../data/Stanford_OCR",load_to_memory=False,load_as_images=True)
        train = data['train']
        test = data['test']
        valid = data['valid']
        print(train[1])
        print(valid[1])
        print(test[1])

        count = 0
        trainImages = []
        trainLabels = []
        trainIter = train[0].__iter__()
        for example in trainIter:
            # example[0] = image array
            # example[1] = image label (int)
            print("Processing Image ", count+1)
            im, label = getImageAndLabelFromMLPythonExample(example, save_image=False, center_images=CENTER_IMAGES)
            trainImages.append(im)
            trainLabels.append(label)
            count+=1

        count = 0
        testImages = []
        testLabels = []
        testIter = test[0].__iter__()
        for example in testIter:
            # example[0] = image array
            # example[1] = image label (int)
            print("Processing Image ", count+1)
            im, label = getImageAndLabelFromMLPythonExample(example, save_image=False, center_images=CENTER_IMAGES)
            testImages.append(im)
            testLabels.append(label)
            count+=1

        # additionalData, additionalLabels = loadAdditionalData(ALPHABET_DATA_PATH, save_images=True, center_images=CENTER_IMAGES)
        additionalData, additionalLabels = loadFormattedCustomAlphabet(ALPHABET_DATA_PATH)

        trainX = [np.array(image) for image in trainImages]
        trainY = makeOneHotVectors(trainLabels)
        testX = [np.array(image) for image in testImages]
        testY = makeOneHotVectors(testLabels)

        additionalY = makeOneHotVectors(additionalLabels)

        with open('../data/data.pkl', 'w') as output:
            dataO = Data(trainX, trainY, testX, testY)
            dataO.balanceDataClasses()
            dataO.addAdditionData(additionalData, additionalY)
            dataO.shuffleTrainData()
            pickle.dump(dataO, output, pickle.HIGHEST_PROTOCOL)
            del dataO

        print "################################"
        print "Done in data.py"
        print "################################"


if __name__ == "__main__":
    main()