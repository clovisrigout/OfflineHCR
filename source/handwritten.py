from PIL import Image, ImageOps
import numpy as np
import glob
import cPickle as pickle #to save object to disk
import random
import os
DATA_DIR = '../data/handwritten_data/'
np.set_printoptions(threshold=np.inf)

alphabet = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z"]

alphabetDict = {"a":0,"b":1,"c":2,"d":3,"e":4,"f":5,"g":6,"h":7,"i":8,"j":9,"k":10,"l":11,"m":12,"n":13,"o":14,"p":15,"q":16,"r":17,"s":18,"t":19,"u":20,"v":21,"w":22,"x":23,"y":24,"z":25}

IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28


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

def invertColor(image):
    return ImageOps.invert(image)

def imageToData(image):
    image = setWhiteBackground(image)                       # set white background
    image = cropImage(image)                                # crop
    image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT), Image.ANTIALIAS)  # resize
    m = makeBinaryMatrixFromImage(image)                    # make binary image: 1=black
    image = Image.fromarray(m)
    image = invertColor(image)
    m = np.array(image)
    return m, image

def makeBinaryMatrixFromImage(image):
    bwImage = image.convert('1')                            # convert to black and white
    A = np.array(bwImage)                                   
    bwA = np.empty((A.shape[0], A.shape[1]), None)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i][j] == True:
                bwA[i][j] = 0
            else:
                bwA[i][j] = 255
    return bwA.astype('uint8')

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
    # A = np.array(image)
    # for x in xrange(A.shape[0]):
    #     for y in xrange(A.shape[1]):
    #         if A[x][y][3] < 10 :
    #             A[x][y] = [255,255,255,255]
    # image = Image.fromarray(A)
    # return image
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


def is_formatted(fileName):
	return not fileName.startswith("1")

def is_image(fileName):
	return fileName.split(".")[1] == "png"

def is_dir(dirName):
	return len(dirName.split(".")) == 1

def loadCustomAlphabet(dir_path, save_images=False, center_images=False):

	count = 0
	for sentDirName in os.listdir(DATA_DIR):
		if(is_dir(sentDirName)):
			print sentDirName
			for wordDirName in os.listdir(os.path.join(dir_path, sentDirName)):
				if(is_dir(wordDirName)):
					print wordDirName
					index = 0
					wordDirPath = os.path.join(os.path.join(dir_path, sentDirName),wordDirName)
					for fileName in os.listdir(wordDirPath):
						if(is_image(fileName) and not is_formatted(fileName)):
							print(fileName)
							fullpath = os.path.join(os.path.join(os.path.join(dir_path, sentDirName),wordDirName),fileName)
							image = Image.open(fullpath)
							imageData, im = imageToData(image)
							if(save_images):
								print "saving image in dir {} to name {}".format(wordDirPath, wordDirName[index])
								newFileName = wordDirName[index]+"_{}.png".format(count)
								count += 1
								savePath = wordDirPath+"/"+newFileName
								if(center_images):
									im = recenterImage(im)
									im = im.convert("RGB").save(savePath)
								else:
									im = im.convert("RGB").save(savePath)
							index += 1


def main():
	loadCustomAlphabet(DATA_DIR, save_images=True)


if __name__ == "__main__":
    main()