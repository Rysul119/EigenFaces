#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 21:01:59 2021

@author: rysul
"""
import cv2 as cv


img = cv.imread("faces/zhixiang-chen.jpg")
# grab the dimensions of the image and calculate the center of the
# image

(h, w) = img.shape[:2]
(cX, cY) = (w // 2, h // 2)
# rotate our image by 45 degrees around the center of the image
M = cv.getRotationMatrix2D((cX, cY), 0, 1.0)
img = cv.warpAffine(img, M, (w, h))
'''
cv.imshow("cropped", img)
cv.waitKey(0)
'''

y = 40
x = 42
crop_img = img[y:y+80, x:x+60]
cv.imwrite('preprocessedFaces/zhixiang-chen'+'cropped.jpg', crop_img)
cv.imshow("cropped", crop_img)
cv.waitKey(0)
