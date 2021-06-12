#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 14:49:44 2021

@author: rysul
"""
import numpy as np
import cv2 as cv
import json
import glob
from matplotlib import pyplot as plt
'''
img = cv.imread('remoteControl.jpg',0)
# Initiate ORB detector
orb = cv.ORB_create()
# find the keypoints with ORB
kp = orb.detect(img,None)
# compute the descriptors with ORB
kp, des = orb.compute(img, kp)
print(des.shape)
print(des)
# saving the descriptors 
#np.savetxt('remote.orb', des, delimiter=',', fmt='%d') 
# loading the descriptors
desCheck = np.loadtxt('remote.orb', delimiter=',', dtype = int)
print(desCheck.shape)
print(desCheck)
# draw only keypoints location,not size and orientation
img2 = cv.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
plt.imshow(img2), plt.show()
'''
# how to compare the feature vectors of the object to be detected with the descriptors of the data base
# probably calculate based on the number of matches. highest match then that's out object

'''
# reshape the images to a same size
img = cv.imread('key.jpg')
imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
imgGrayResized = cv.resize(imgGray, (800,600), interpolation = cv.INTER_AREA)
cv.imwrite('keyResizedGray.jpg',imgGrayResized)
'''
'''
# create the orb data files and checking their shapes
img = cv.imread('keyMine.jpg')
imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
imgGrayResized = cv.resize(imgGray, (800,600), interpolation = cv.INTER_AREA)
# Initiate ORB detector
orb = cv.ORB_create()
# compute the descriptors and keypoints with ORB
kp, des = orb.detectAndCompute(imgGrayResized,None)
print(des.shape)
print(des)
# saving the descriptors 
np.savetxt('keyMineResizedGray.orb', des, delimiter=',', fmt='%d') 
'''



img = cv.imread('keyMineWithPen.jpg')
imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
imgGrayResized = cv.resize(imgGray, (800,600), interpolation = cv.INTER_AREA)
# img1 = cv.imread('watchResizedGray.jpg',cv.IMREAD_GRAYSCALE)          # queryImage
#img2 = cv.imread('remoteControl.jpg',cv.IMREAD_GRAYSCALE) # trainImage
orb = cv.ORB_create()
# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(imgGrayResized,None)

# using number of matches is not a robust matching/detecting criteria till now.
# takes minimums of all the matches. Then index of minimum of the minimums will correspond to the specific orb feature.

# loading the descriptors
orbFeatureFiles = ['keyMine', 'watch', 'remoteControl']
objects = ['Key', 'Watch', 'Remote Control']
orbMatches = []

for name in orbFeatureFiles:
    desCheck = np.loadtxt(name+'ResizedGray.orb', delimiter=',', dtype = np.uint8) # loading should be in np.unit8
    # create BFMatcher object
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    #print(des1.shape)
    #print(desCheck.shape)
    # Match descriptors.
    matches = bf.match(des1,desCheck)

    #print(len(matches))

    matchDistances=[]
    for match in matches:
        matchDistances.append(match.distance)

    #print(min(matchDistances))
    orbMatches.append(min(matchDistances))
print(orbMatches)  
print(orbMatches.index(min(orbMatches)))


print('\nThere is a '+objects[orbMatches.index(min(orbMatches))]+' in the scene.')


# finally drawing the matched key points with the given object in the scene.

# two modes: 1) create own orb files, 2) already existing orb detection


'''
class for object detection in a scene. 
the class will include function for 
acquisition, 
preprocessing (lecture video, resize),
training
classification
accuracy calculation
doing these and implement tkinter to facilitate this (uploading an image, naming the object, creating orb file, classification)
'''

def loadFiles(path, mode = 'orb'):
    '''
    loads all the files within that path (for getting all the orb files and the query images)
    returns a list with the orb file descriptors or image arrays
    '''
    fileContents = []
    
    for fileName in glob.glob(path):
        if (mode == 'image'):
            img = cv.imread(fileName)
            imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            fileContent = cv.resize(imgGray, (800,600), interpolation = cv.INTER_AREA)
        elif (mode == 'orb'):
            fileContent = np.loadtxt(fileName, delimiter=',', dtype = np.uint8) # loading should be in np.unit8

        fileContents.append(fileContent)
    
    return fileContents


class objectDetectionORB():
    def __init__(self):
        self.imgSize = (800, 600)
        self.matchingMetric = cv.NORM_HAMMING
        
    def acquisition(self, pathFilename):
        '''
        takes/upload an image. takes an image of a scene. Upload to create ORB features
        returns a preprocessed image
        '''
        img = cv.imread(pathFilename)
        imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        self.imgGrayResized = cv.resize(imgGray, (800,600), interpolation = cv.INTER_AREA)
        
    
    def training(self, pathFilename, objectName):
        '''
        capture or upload an image using acquisition. 
        Then create orb features an save it as a .orb file
        '''
        self.acquisition(pathFilename)
        # Initiate ORB detector
        orb = cv.ORB_create()
        # compute the descriptors and keypoints with ORB
        kp, des = orb.detectAndCompute(self.imgGrayResized,None)
        #print(des.shape)
        #print(des)
        # saving the descriptors 
        np.savetxt('objectFeatures/'+objectName+'.orb', des, delimiter=',', fmt='%d')
        
        # adding the object name in a json file
        # load the json file and append the new object name
        with open("objects.json", 'r') as f:
            objects = json.load(f)
            objects.append(objectName)
        # writing the new object name in the object json file
        with open("objects.json", 'w') as f:
           json.dump(objects, f, indent=2)
        
    def classification(self, pathFilename):
        '''
        capture or upload an image to classify if there is any familiar object in the scene
        takes minimums of all the matches. Then index of minimum of the minimums will correspond to the specific orb feature.
        '''
        
        self.acquisition(pathFilename)
        orb = cv.ORB_create()
        
        # find the keypoints and descriptors with ORB
        kp, des = orb.detectAndCompute(self.imgGrayResized,None)
        
       
        # get the object names with orb feature files
        with open("objects.json", 'r') as f:
            featureObjects = json.load(f)
    
        
        # loading all the orb descriptor files
        orbFeatures = loadFiles("objectFeatures/*")
      
        orbMatches = []
        
        for desCheck in orbFeatures:
            # create BFMatcher object
            bf = cv.BFMatcher(self.matchingMetric, crossCheck=True)
            #print(des1.shape)
            #print(desCheck.shape)
            # Match descriptors.
            matches = bf.match(des,desCheck)
        
            #print(len(matches))
        
            matchDistances=[]
            for match in matches:
                matchDistances.append(match.distance)
        
            #print(min(matchDistances))
            orbMatches.append(min(matchDistances))
            
        print(orbMatches.index(min(orbMatches)))
        
        
        
        print('\nThere is a '+featureObjects[orbMatches.index(min(orbMatches))]+' in the scene.')
        
        
        def visualization(self):
            '''
            drawing the matched key points with the given object in the scene.
            '''
            pass


# TODO priority: Integrate with Tkinter << done
# create image dataset with image arrays in a list along with the labels. Labels will be done by hand


'''
dataset labels:
    key: 0
    watch: 1
    caclulator: 2
    phone: 3
'''

# accuracy calculation from the files in imageDataset

'''
# adding the object name in a json file
# load the json file 
labels = [0 , 0]
with open("labels.json", 'r') as f:
    objects = json.load(f)
    
# writing labels in the labels json file by appending the labels for corresposding images
with open("labels.json", 'w') as f:
    for label in labels:
        objects.append(label)
    json.dump(objects, f, indent=2)
'''

def accuracyCalc(): 
    # load the labels
    
    with open("labels.json", 'r') as f:
        flabels = json.load(f)
        
    # load all the images
    
    imageData = loadFiles('imageDataset/*', mode = 'image')
    
    preLabels = []
    
    orbFeatureFiles = ['keyMine', 'watch', 'remoteControl']
    objects = ['Key', 'Watch', 'Remote Control']
    
    for image in imageData:
        orb = cv.ORB_create()
        # find the keypoints and descriptors with ORB
        kp1, des1 = orb.detectAndCompute(image,None)
        
        # loading the descriptors
        
        orbMatches = []
        
        for name in orbFeatureFiles:
            desCheck = np.loadtxt(name+'ResizedGray.orb', delimiter=',', dtype = np.uint8) # loading should be in np.unit8
            # create BFMatcher object
            bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
            #print(des1.shape)
            #print(desCheck.shape)
            # Match descriptors.
            matches = bf.match(des1,desCheck)
        
            #print(len(matches))
        
            matchDistances=[]
            for match in matches:
                matchDistances.append(match.distance)
        
            #print(min(matchDistances))
            orbMatches.append(min(matchDistances))
            
        
        print(orbMatches.index(min(orbMatches)))
        
        preLabels.append(orbMatches.index(min(orbMatches)))
        
        print('\nThere is a '+objects[orbMatches.index(min(orbMatches))]+' in the scene.')
    
    accuracy = sum(1 for x,y in zip(flabels,preLabels) if x == y) / len(flabels)
    return accuracy * 100

accuracy = accuracyCalc()
print('Accuracy is: {}%'.format(accuracy))
        
# accuracy + training
# clean it up as much as i can
# start the report


