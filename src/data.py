
###############################################################################################
####################################### CUSTOM CODE ###########################################
###############################################################################################
import sys 
sys.path.append('..')

import numpy as np
np.set_printoptions(threshold=np.inf)

import os
from PIL import Image, ImageOps
import glob
import cPickle as pickle #to save object to disk
import random

from stanford_ocr import obtain, load


alphabet = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"]

alphabetDict = {"a":0,"b":1,"c":2,"d":3,"e":4,"f":5,"g":6,"h":7,"i":8,"j":9,"k":10,"l":11,"m":12,"n":13,"o":14,"p":15,"q":16,"r":17,"s":18,"t":19,"u":20,"v":21,"w":22,"x":23,"y":24,"z":25}

IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28

EXTRA_GENERATED = 3000           #number of images generated per class

DOWLOAD_DATA = True              #flag which downloads stanford ocr dataset
FORMAT_PAINT_ALPHABET = False    #flag converts paint data to formatted data
CREATE_DATA_OBJECT = True        #flag which creates the data object used by model
CENTER_IMAGES = False            #flag to center images
CUSTOM_GENERATION = True         #flag to generate custom data
SAVE_SAMPLES = True              #flag which saves samples of object data

DATA_OBJECT_PATHS = {True:'../data/dataWithGen.pkl', False: '../data/data.pkl'}
ALPHABET_DATA_PATH = "../data/paint_data"
FORMATTED_ALPHABET_DATA_PATH = "../data/formatted_data"
HANDWRITING_DATA_PATH = "../data/handwritten_data"
SAMPLE_DATA_PATH = "../data/samples/"


##################################### DATA OBJECT ########################################

class Data:
    def __init__(self,trainX, trainY, testX, testY):
        self.trainX = trainX
        self.trainY = trainY
        self.testX = testX
        self.testY = testY
        self.test = [testX, testY]
        self.alphabetX = []
        self.alphabetY = []
        self.handwritingX = []
        self.handwritingY = []
        self.customX = []
        self.customY = []
        self.words = []
        self.batchIndex = 0

    def addAlphabet(self, alphabetX, alphabetY):
        self.alphabetX = alphabetX
        self.alphabetY = alphabetY

    def addAdditionalData(self, additionalX, additionalY):
        self.trainX.extend(additionalX)
        self.trainY.extend(additionalY)

    def addCustomData(self, customX, customY):
        self.customX = customX
        self.customY = customY

    def addHandwritingData(self, handwritingX, handwritingY):
        self.handwritingX = handwritingX
        self.handwritingY = handwritingY

    def addWordsData(self, words):
        self.words = words

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


 ####################################### MAIN METHOD########################################

def main():
    if(CREATE_DATA_OBJECT):
        if(DOWLOAD_DATA):
            obtain("./Stanford_OCR") #download dataset
        data = load("../data/Stanford_OCR",load_to_memory=False,load_as_images=True)
        train = data['train']
        test = data['test']
        valid = data['valid']

        trainX, trainY = getMLData(train[0].__iter__())

        testX, testY = getMLData(test[0].__iter__())

        if(FORMAT_PAINT_ALPHABET):
            alphabetX, alphabetLabels = loadCustomAlphabet(ALPHABET_DATA_PATH, save_images=True, center_images=CENTER_IMAGES)
            alphabetY = makeOneHotVectors(alphabetLabels)
        else:
            alphabetX, alphabetLabels = loadFormattedCustomAlphabet(FORMATTED_ALPHABET_DATA_PATH)
            alphabetY = makeOneHotVectors(alphabetLabels)

        handwritingX, handwritingLabels, words = loadFormattedCustomHandwriting(HANDWRITING_DATA_PATH)
        handwritingY = makeOneHotVectors(handwritingLabels)

        # creating and saving data object
        with open(DATA_OBJECT_PATHS[CUSTOM_GENERATION], 'w') as output:
            dataO = Data(trainX, trainY, testX, testY)
            dataO.balanceDataClasses()
            dataO.addAlphabet(alphabetX, alphabetY)
            customData, customLabels = generateCustomData(dataO.alphabetX, dataO.alphabetY)
            dataO.addAdditionalData(customData, customLabels) # add custom data to training data
            dataO.addCustomData(customData, customLabels)
            dataO.shuffleTrainData()
            dataO.addHandwritingData(handwritingX, handwritingY)
            pickle.dump(dataO, output, pickle.HIGHEST_PROTOCOL)
            del dataO

        print "################################"
        print " Created data object at path {}".format(DATA_OBJECT_PATHS[CUSTOM_GENERATION])
        print "################################"

    else: 
        path = DATA_OBJECT_PATHS[not CUSTOM_GENERATION] #open non-custom data
        print("loading data at path {}".format(path))

        with open(path, 'rb') as input:
            dataO = pickle.load(input)

            if(CUSTOM_GENERATION):
                customData, customLabels = generateCustomData(dataO.alphabetX, dataO.alphabetY)
                dataO.addAdditionalData(customData, customLabels) # add custom data to training data
                dataO.addCustomData(customData, customLabels)
                dataO.shuffleTrainData()

                with open(DATA_OBJECT_PATHS[CUSTOM_GENERATION], 'w') as output:
                    pickle.dump(dataO, output, pickle.HIGHEST_PROTOCOL)
                    print "saved at {}".format(DATA_OBJECT_PATHS[CUSTOM_GENERATION])

                print "######################ADDED CUSTOM DATA############################"

    if(SAVE_SAMPLES):
        trainExample = Image.fromarray(dataO.trainX[0])
        testExample = Image.fromarray(dataO.testX[0])
        alphabetExample = Image.fromarray(dataO.alphabetX[0])
        handwritingExample = Image.fromarray(dataO.handwritingX[0])
        customExample = Image.fromarray(dataO.customX[0])
        r = performRandomTransformation(customExample)

        trainExample.save(SAMPLE_DATA_PATH+"trainExample.png")
        testExample.save(SAMPLE_DATA_PATH+"testExample.png")
        alphabetExample.save(SAMPLE_DATA_PATH+"alphabetExample.png")
        handwritingExample.save(SAMPLE_DATA_PATH+"handwritingExample.png")
        customExample.save(SAMPLE_DATA_PATH+"customExample.png")
        r.save(SAMPLE_DATA_PATH+"random.png")


    print "\n\n################################"
    print "Done in data.py"
    print "################################"


##################################### HELPER METHODS ########################################


def cropImage(image, mode="RGBA"):
    imD = image.load()
    (X, Y) = image.size
    m = np.zeros((X, Y))

    topIndex = Y
    leftIndex = X
    rightIndex = 0
    bottomIndex = 0

    if(mode == "L"):
        entry = [255]
    elif(mode == "RGB"):
        entry = [(255,255,255)]
    elif(mode == "RGBA"):
        entry = [(255,255,255,255)]
    else:
        print "ERROR IN CROP IMAGE, UNSUPPORTED MODE"

    for x in range(X):   # x is column index
        col = [imD[(x,row)] for row in xrange(Y)]
        if(col != entry*Y):
            if(leftIndex) > x:
                leftIndex = x
            if(rightIndex) < x:
                rightIndex = x
    for y in range(Y):
        row = [imD[(col,y)] for col in xrange(X)]
        if(row != entry*X):
            if(topIndex) > y:
                topIndex = y
            if(bottomIndex) < y:
                bottomIndex = y

    image = image.crop((max(0,leftIndex-1), max(0,topIndex-1), min(rightIndex+1,X), min(bottomIndex+1,Y)))
    return image


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


def generateCustomData(alphabetX, alphabetY):
    customData = []
    customLabels = []
    for letterD, label in zip(alphabetX, alphabetY):
        newData = generateTransformedData(letterD, EXTRA_GENERATED)
        newLabels = [label for d in newData]
        customData.extend(newData)
        customLabels.extend(newLabels)
    return customData, cus


def generateTransformedData(data, count):
    transformedData = []
    im = Image.fromarray(data)
    while(count > 0):
        transformedImage = performRandomTransformation(im).convert("L")
        # transformedD = imageToData(transformedImage)
        transformedD = np.array(transformedImage)
        transformedData.append(transformedD)
        count = count-1
    return transformedData


def getImageAndLabelFromMLPythonExample(example, save_image=False, center_images=False):
    # example[0] = image array
    # example[1] = image label (int)
    im = Image.fromarray(np.array(example[0]))
    im = invertColor(im).convert("L")
    # im = im.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.ANTIALIAS)
    imD = makeBinaryMatrixFromImage(im)
    im = Image.fromarray(imD)
    im = cropImage(im, mode=im.mode)
    im = im.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.ANTIALIAS)
    imD = makeBinaryMatrixFromImage(im)
    im = Image.fromarray(imD)
    if(center_images):
        im = recenterImage(im).convert("L")
    if(save_image):
        im.save("../data/Stanford_OCR/images/"+alphabet[int(example[1])]+".png")
    return im, alphabet[int(example[1])]


def getMLData(iterator):
    count = 0
    images = []
    labels = []
    for example in iterator:
        # example[0] = image array
        # example[1] = image label (int)
        print("Processing Image ", count+1)
        im, label = getImageAndLabelFromMLPythonExample(example, save_image=True, center_images=CENTER_IMAGES)
        images.append(im)
        labels.append(label)
        count+=1

    X = [np.array(image.convert("L")) for image in trainImages]
    Y = makeOneHotVectors(labels)
    return X, Y


    
def imageToData(image):
    image = setWhiteBackground(image)                       # set white background
    image = cropImage(image)                                # crop
    image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.ANTIALIAS)  # resize
    m = makeBinaryMatrixFromImage(image)                    # make binary image: 0=black
    image = Image.fromarray(m).convert("L")
    m = np.array(image)
    return m, image


def invertColor(image):
    return ImageOps.invert(image)


# used to format the initial paint data alphabet
def loadCustomAlphabet(dir_path, save_images=False, center_images=False):
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
            savePath = "../data/formatted_data/"+fileName+".png"
            if(center_images):
                im = recenterImage(im)
                im = im.convert("RGBA").save(savePath)
            else:
                im = im.convert("RGBA").save(savePath)
    return data,labels

# used once paint data alphabet has already been formatted
def loadFormattedCustomAlphabet(dir_path):
    dir_path = os.path.expanduser(dir_path)

    data = []
    labels = []

    for filePath in glob.glob(dir_path +"/*.png"):
        fileName = filePath.split(".png")[0].split("/")[-1]
        image = Image.open(filePath)
        imageData = makeBinaryMatrixFromImage(image)               # make binary image: 0=black
        im = Image.fromarray(imageData).convert("L")
        imageData = np.array(im)
        data.append(imageData)
        labels.append(fileName)
    return data,labels


def is_formatted(fileName):
    return not fileName.startswith("1")

def is_image(fileName):
    return fileName.split(".")[1] == "png"

def is_dir(dirName):
    return len(dirName.split(".")) == 1

# loads already formatted target handwriting test set
def loadFormattedCustomHandwriting(dir_path):
    data = []
    words = []
    labels = []
    for sentDirName in os.listdir(dir_path):
        if(is_dir(sentDirName)):
            for wordDirName in os.listdir(os.path.join(dir_path, sentDirName)):
                if(is_dir(wordDirName)):
                    wordDirPath = os.path.join(os.path.join(dir_path, sentDirName),wordDirName)
                    word_temp = {}
                    for fileName in os.listdir(wordDirPath):
                        if(is_image(fileName) and is_formatted(fileName)):
                            fullpath = os.path.join(os.path.join(os.path.join(dir_path, sentDirName),wordDirName),fileName)
                            image = Image.open(fullpath)
                            imageData = makeBinaryMatrixFromImage(image)
                            image = Image.fromarray(imageData).convert("L")
                            imageData = np.array(image)
                            data.append(imageData)
                            labels.append(fileName.split("_")[0])
                            word_temp[fileName.split("_")[0]] = imageData
                    word = []
                    for index, c in enumerate(wordDirName.split("_")[0]):
                        word.append((word_temp[c], c))
                    words.append(word)
    return data, labels, words


def makeBinaryMatrixFromImage(image):
    bwImage = image.convert('1')                            # convert to black and white
    A = np.array(bwImage)                                   
    bwA = np.empty((A.shape[0], A.shape[1]), None)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i][j] == True:
                bwA[i][j] = 255
            else:
                bwA[i][j] = 0
    return bwA.astype('uint8')

def makeOneHotVectors(labels):
    vectors = []
    for label in labels:
        vec = np.zeros(26)
        index = alphabetDict[label]
        vec[index] = 1
        vectors.append(vec)
    return vectors

# Takes an image and performs a random transformation to it. Used to generate random data
def performRandomTransformation(image):
    # transformation is either : stretch or rotate or both
    if image.mode == "L":
        black = 0
    elif image.mode == "RGB":
        black = (0,0,0)
    elif image.mode == "RGBA":
        black = (0,0,0,0)
    else:
        "ERROR IN performRandomTransformation, UNSUPPORTED mode"
        return
    image = invertColor(image)
    canvas = Image.new(image.mode, (image.size[0]*2,image.size[1]*2), black)
    canvas.paste(image, (canvas.size[0]/2, canvas.size[0]/2))
    image = canvas
    transformType = random.randint(1,3)
    if(transformType == 1): # stretch / shrink
        a = float(random.randint(100,300))/float(100)
        e = float(random.randint(100,300))/float(100)
        transformed = image.transform(image.size, Image.AFFINE, (a,0,0,0,e,0))
        transformed = invertColor(transformed)
        transformed = cropImage(transformed, mode=transformed.mode).resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.ANTIALIAS)
        transformedD = makeBinaryMatrixFromImage(transformed)
        transformed = Image.fromarray(transformedD)
        return transformed
    elif(transformType == 2): #rotate
        rotation = random.randint(-25,25)
        transformed = image.rotate(rotation)
        transformed = invertColor(transformed)
        transformed = cropImage(transformed, mode=transformed.mode).resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.ANTIALIAS)
        transformedD = makeBinaryMatrixFromImage(transformed)
        transformed = Image.fromarray(transformedD)
        return transformed
    else:
        a = float(random.randint(100,300))/float(100)
        e = float(random.randint(100,300))/float(100)
        transformed = image.transform(image.size, Image.AFFINE, (a,0,0,0,e,0))
        rotation = random.randint(-25,25)
        transformed = transformed.rotate(rotation)
        transformed = invertColor(transformed)
        transformed = cropImage(transformed, mode=transformed.mode).resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.ANTIALIAS)
        transformedD = makeBinaryMatrixFromImage(transformed)
        transformed = Image.fromarray(transformedD)
        return transformed


def recenterImage(image):
    (cx, cy) = findCenterOfMass(image)
    im = shiftImageCenterToPoint(image, (int(cx), int(cy)), (int(IMAGE_WIDTH/2),int(IMAGE_HEIGHT/2)))
    return im

def setWhiteBackground(image):
    canvas = Image.new('RGBA', image.size, (255,255,255,255))
    canvas.paste(image, mask=image)
    return canvas

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

if __name__ == "__main__":
    main()