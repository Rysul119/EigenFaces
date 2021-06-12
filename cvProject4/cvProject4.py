#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 14:34:50 2021

@author: rysul
"""

import numpy as np
import cv2 as cv
import json
import glob



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
        
       
    
        
        # loading all the orb descriptor files
        orbFeatures, featureObjects = loadFiles("objectFeatures/*")
      
        orbMatches = [] # to store all the corresponding minimum match distances for each orb feature file
        
        for desCheck in orbFeatures:
            # create BFMatcher object
            bf = cv.BFMatcher(self.matchingMetric, crossCheck=True)
            #print(des1.shape)
            #print(desCheck.shape)
            # Match descriptors.
            matches = bf.match(des,desCheck)
        
            #print(len(matches))
            
            matchDistances=[] # to store all the match distances
            for match in matches:
                matchDistances.append(match.distance)
        
            #print(min(matchDistances))
            orbMatches.append(min(matchDistances))
            
        print(orbMatches.index(min(orbMatches)))
        
        
        print('\nThere is a '+featureObjects[orbMatches.index(min(orbMatches))]+' in the scene.')
        
        
        
def loadFiles(path, mode = 'orb'):
    '''
    loads all the files within that path (for getting all the orb files and the query images)
    returns a list with the orb file descriptors or image arrays
    '''
    fileContents = []
    orbNames = []
    for fileName in glob.glob(path):
        if (mode == 'image'):
            img = cv.imread(fileName)
            imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            fileContent = cv.resize(imgGray, (800,600), interpolation = cv.INTER_AREA)
        elif (mode == 'orb'):
            fileContent = np.loadtxt(fileName, delimiter=',', dtype = np.uint8) # loading should be in np.unit8
       
        f = fileName.split('.')[0]
        orbNames.append(f.split('/')[1])
        fileContents.append(fileContent)
    
    return fileContents, orbNames


def accuracyCalc(): 
    '''
    calculates the accuracy on the existing image dataset for detecting objects using ORB
    '''
    # load the labels
    with open("labels.json", 'r') as f:
        flabels = json.load(f)
        
    # load all the images
    
    imageData, imageNames = loadFiles('imageDataset/*', mode = 'image')
    
    # arranging the labels according to the loaded images
    flabelsM = [flabels[int(i)-1] for i in imageNames]
    
    preLabels = []
    
    
    for image in imageData:
        orb = cv.ORB_create()
        # find the keypoints and descriptors with ORB
        kp1, des1 = orb.detectAndCompute(image,None)
        
        
        # loading all the orb descriptor files
        orbFeatures, featureObjects = loadFiles("objectFeatures/*")
        
        orbMatches = []
        
        for desCheck in orbFeatures:
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
        
        obj = featureObjects[orbMatches.index(min(orbMatches))]
        #print('\nThere is a '+obj+' in the scene.')
        
        '''
        dataset labels:
            key: 0
            watch: 1
            caclulator: 2
            phone: 3
        '''
        if(obj=='key'):
            preLabels.append(0)
        elif(obj=='watch'):
            preLabels.append(1)
        elif(obj=='calculator'):
            preLabels.append(2)
        elif(obj=='phone'):
            preLabels.append(3)
    
    accuracy = sum(1 for x,y in zip(flabelsM,preLabels) if x == y) / len(flabelsM)
    return accuracy * 100       


if __name__ == "__main__":
    
    # initiates the ObjectDetectiongOrb
    objectDetection = objectDetectionORB()
    
    # creates orb file with a corresponding object name
    #objectDetection.training('images/phone.jpg', 'phone')
    
    # classifies a scene
    #objectDetection.classification('images/keyRotate.jpg')
    
    
 
    '''    
    # adding the new image dataset object labes in a json file
    # load the json file 
    labels = [1,3,2,3,1,2]
    with open("labels.json", 'r') as f:
        objects = json.load(f)
        
    # writing labels in the labels json file by appending the labels for corresposding images
    with open("labels.json", 'w') as f:
        for label in labels:
            objects.append(label)
        json.dump(objects, f, indent=2)
    '''
    
    # calculates the accuracy of the orb implementation
    accuracy = accuracyCalc()
    print('\nAccuracy of this ORB implementation for object detection is: {}%.'.format(accuracy))
        

    # clean the GUI.
    # create the report part except overview of ORB but with flowchart and results.

        
        
        
        
        
        