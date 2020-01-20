# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 01:09:07 2019

@author: Admin
"""

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import dlib
import imutils



ksize=50 #Kernal Size
sigma=3
theta=1*np.pi/8
lamda=1*np.pi/4 

gamma=1
phi=1 #offset

kernel=cv.getGaborKernel((ksize,ksize),sigma,theta,lamda,gamma,phi,ktype=cv.CV_32F)
plt.imshow(kernel)
plt.show


img=cv.imread('PPB/0009.jpg')
img=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    
detector = dlib.get_frontal_face_detector()
rects = detector(img, 1)    #Detect Face

shape = predictor(img, rects)


img=dlib.drectangle(img,rec)


(x,y,w,h)=imutils.face_utils.rect_to_bb(rec)
 
predictor = dlib.shape_predictor(img)


cv.imshow('Original Image',img)
cv.waitKey()
cv.destroyAllWindows()



fimg=cv.filter2D(img, cv.CV_8UC3, kernel )
kernel_resize=cv.resize(kernel,(800,800))

cv.imshow('Original Image',img)
cv.imshow('Filtered Image',fimg)
cv.imshow('Kernel',kernel)
cv.waitKey()
cv.destroyAllWindows()
'-------------------------------------------------------------------------------------------------

# SELECT LAND MARKS (LANDMARK POINT )
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('D:/DATA SET FOR DARK SKINE/PPB/Code/FR_PPB/Classifier/dlib/shape_predictor_68_face_landmarks.dat')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
rects = detector(gray, 1)
for (i, rect) in enumerate(rects):
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)
    p=[36,39,42,45,33,48,54,8]
    for i in p:
        (x,y)=shape[i]
        jt(img,x,y,5)
#        cv2.circle(img, (x, y), 1, (0, 0, 255), -1)

# EXTRACT JET AT EACH LANGMARK
def jt(img,x,y,m):
    cv2.rectangle(img,(x-m,y-m),(x+m,y+m),(0, 255, 0),1)
# IMAGE TETST
cv2.imshow('image',img)
cv2.waitKey()
cv2.destroyAllWindows()

import random
s=random.sample(range(0,len(klist)),30)
for i in range(len(s)):
    img = klist[i]
    plt.subplot(10,3,1+i)
    plt.imshow(img, cmap='gray')
    plt.tick_params(labelleft='off', labelbottom='off', bottom='off',top='off',right='off',left='off', which='both')
plt.show()        

plt.imshow(img)
plt.show()
