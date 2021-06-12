#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 19:53:31 2021

@author: rysul
"""

import cv2 as cv
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math
from sys import getsizeof

class EigenFaces():
    def __init__(self, path):
        
        self.imagePath = path
    
    def loadFiles(self):
        '''
        loads all the image files within that path
        returns 
        a horizontally stacked array of each of the flattened image array
        a list of  
        
        '''
        totImgSize = 0
        fileContents = []
        fileNames = []
        for fileName in glob.glob(self.imagePath):
            img = cv.imread(fileName, cv.IMREAD_GRAYSCALE)
            totImgSize += getsizeof(img)
            self.row, self.column = img.shape 
            img = img.flatten()
            fileContent = img[:, np.newaxis]
            fileContents.append(fileContent)
            f = fileName.split('.')[0]
            f = f.split('/')[1]
            f = f.split('Resized')[0]
            fileNames.append(f)
        
            
        print("Total loaded image size {} bytes".format(totImgSize))
        
        stackedArray = np.hstack(fileContents)
            
        return stackedArray, fileContents, fileNames
    
    def getPCA(self):
        
        '''
        obtains the sorted eigenvectors and eigenvalues
        '''
        
        imgArray, self.imgList, self.imgNames = self.loadFiles()
        print("Image shapes {} x {}".format(self.row, self.column))
        print("Data shape {}".format(imgArray.shape))
        
        # centering and normalizing the data
        '''
        getting the mean  
        and normalizing
        '''
        self.meanImage = np.sum(imgArray, axis = 1)[:, np.newaxis]
        self.meanImage = (self.meanImage/len(self.imgList))
        meanImgaeSize = getsizeof(self.meanImage.reshape(self.row, self.column))
        print("Mean image size {} bytes".format(meanImgaeSize))
        
        # centered data no normalization
        self.normsD = np.linalg.norm(imgArray, axis = 0)
        self.imgArrayCentered = imgArray - self.meanImage
        self.imgArrayCentered /= self.normsD
        
        print("Centered data shape {}".format(self.imgArrayCentered.shape))
        
        # covariance matrix
        covMat = np.matmul(self.imgArrayCentered.T, self.imgArrayCentered)
        covMat /= len(self.imgList)
        
        print("Covariance matrix shape {}".format(covMat.shape))
        
        
        # get the eigen values and eigen vectors
        self.eigenVals, self.eigenVecs = np.linalg.eig(covMat)
        
        #sorting decresing order the eigenVals and corresponsing eigenvectors
        sortIndices = self.eigenVals.argsort()[::-1]
        self.eigenVals = self.eigenVals[sortIndices]
        self.eigenVecs = self.eigenVecs[:, sortIndices]
        
    def variableExplained(self, varThreshold):
        
        self.varThreshold = varThreshold
        self.getPCA()
        valSum = np.sum(self.eigenVals)
        self.count = 0
        varExplained = 0
        for eigenVal in self.eigenVals:
            self.count += 1
            varExplained += eigenVal/valSum
            if (varExplained > varThreshold):
                break
        
        self.eigenVals = self.eigenVals[0:self.count]
        self.eigenVecs = self.eigenVecs[:, 0:self.count]
        print("Eigen value shape {}".format(self.eigenVals.shape))
        print("Eigen vector shape {}".format(self.eigenVecs.shape))
    
    
    def getEigenFaces(self):
        # U
        self.u = np.matmul(self.imgArrayCentered, self.eigenVecs)
        norms = np.linalg.norm(self.u, axis = 0)
        self.u /= norms
        print("U shape {}".format(self.u.shape))
        
        # view eigen faces
        self.eigenFaces = []
        for i in range(self.count):
            eigenFace = self.u[:,i].reshape(self.row,self.column)
            eigenFace = cv.normalize(eigenFace, 0,255, cv.NORM_MINMAX)
            self.eigenFaces.append(eigenFace) 
     
        sizeOfEigenFaces = getsizeof(self.eigenFaces)
        print("Eigen faces size {} bytes".format(sizeOfEigenFaces))
     
    def showEigenFace(self, count):
        '''
        shows the number of given number of highest eigen faces
        '''
        self.getEigenFaces()
        figEdge = int(np.ceil(math.sqrt(count))) #square format
        plt.figure(figsize = (figEdge, figEdge))
        gs1 = gridspec.GridSpec(figEdge, figEdge)
        gs1.update(wspace=0.025, hspace=0.05) # set the spacing between axes. 
        
        for i in range(count):
            ax1 = plt.subplot(gs1[i])
            plt.axis('off')
            plt.imshow(self.eigenFaces[i],cmap = 'gray')
            ax1.set_aspect('equal')
        plt.savefig('outputs/EigenFaces'+str(count)+'jpg')
        plt.show()
        
        
    def getprojectCoeff(self):
        
        # projection
        self.getEigenFaces()
        self.projAll = np.matmul(self.u.T, self.imgArrayCentered)
        print("Projection coefficient shape {}".format(self.projAll.shape))
        
        projAllSize = getsizeof(self.projAll)
        print("Projection coefficient size {} bytes".format(projAllSize))
    
    def reconstruction(self):
        
        self.getprojectCoeff()
        # reconstruction
        self.reconAll = np.matmul(self.u, self.projAll) * self.normsD + self.meanImage
        #reconAll = np.matmul(u, projAll) + meanImage
        print("Reconstruction shape {}".format(self.reconAll.shape))
    
    def showReconstruction(self, imageIndex):
        
        self.reconstruction()
        # get the original image
        plt.subplot(131)
        ogImage = self.imgList[imageIndex].reshape(self.row,self.column)
        #ogImage = cv.normalize(ogImage, 0,255, cv.NORM_MINMAX)
        plt.imshow(ogImage, cmap = 'gray')
        plt.title('Original')
        plt.axis('off')
        
        plt.subplot(132)
        meanImg = self.meanImage.reshape(self.row,self.column)
        meanImg = cv.normalize(meanImg, 0,255, cv.NORM_MINMAX)
        plt.imshow(meanImg, cmap = 'gray')
        plt.title('Mean Image')
        plt.axis('off')
        
        plt.subplot(133)
        reconFace = self.reconAll[:,imageIndex].reshape(self.row,self.column)
        reconFace = cv.normalize(reconFace, 0,255, cv.NORM_MINMAX)
        plt.imshow(reconFace, cmap = 'gray')
        plt.title('Reconstructed')
        plt.axis('off')
    
        plt.savefig('outputs/reconstruction'+ self.imgNames[imageIndex]+str(self.varThreshold) +'.jpg')

if __name__ == '__main__':
    
    eFaces = EigenFaces('resizedFaces/*')
    
    eFaces.variableExplained(1)

    eFaces.showReconstruction(17)
    
    # shows the first 10 eigenfaces
    #eFaces.showEigenFace(10)
        
