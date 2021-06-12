#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 08:22:33 2021

@author: rysul
"""

import cv2 as cv
import glob 

def loadFiles(path):
    '''
    loads all the image files within that path convert them to gray, resize them, and save them in a .jpg image
    '''
    for fileName in glob.glob(path):
        img = cv.imread(fileName)
        imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        fileContent = cv.resize(imgGray, (125,150), interpolation = cv.INTER_AREA)
        f = fileName.split('.')[0]
        f = f.split('/')[1]
        f = f.split('cropped')[0]
        cv.imwrite('resizedFaces/'+ f + 'Resized.jpg', fileContent)

   

if __name__ == '__main__':

    loadFiles('preprocessedFaces/*')
    
    
    