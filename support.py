# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 18:36:00 2019

@author: Admin
"""

import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import csv
from PIL import Image
import numpy as np
import random
import dlib
from imutils import face_utils




# DATA SET DICTIONARY
def load(lbl_path):
    lbl=lbl_gen(lbl_path)
    raw_d=load_image()
    
    return{'raw_img':raw_d,
           'label':lbl}


# LOAD IMAGE
def load_image():
    rawdata=[]
    for im in range(len([name for name in os.listdir('PPB')])):
        im=im+1
        rawdata.append(cv2.imread('PPB/'+cd(im)+'.jpg'))

    return rawdata



    
# RESIZE IMAGE
def re_size(img_set,x,y):
    dataset=[]
    for im in img_set:
        dataset.append(cv2.resize(im, (x,y)))
    
    return dataset




# FACE DETECTOR
def fc_detect(img_lst):
    protext='Classifier/cafnet/deploy.prototxt.txt'
    weight='Classifier/cafnet/res10_300x300_ssd_iter_140000.caffemodel'
    model = cv2.dnn.readNetFromCaffe(protext, weight)
    crop=[]
    for im in img_lst:
        (h, w) = im.shape[:2]
        im=cv2.resize(im, (300,300))
        blob = cv2.dnn.blobFromImage(im,1.0,(300,300),(104.0, 177.0, 123.0))
        model.setInput(blob)
        detections = model.forward()
        for i in range(0, detections.shape[2]):
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            confidence = detections[0, 0, i, 2]
            if (confidence > 0.165):
                crop.append(im[startY:endY,startX:endX])
    return crop



# CONVERT TO GRAY SCALE
def gray(img_arr):
    gr=[]
    for i in img_arr:
        gr.append(cv2.cvtColor(i,cv2.COLOR_BGR2GRAY))
    return gr




# CROP FACE
def crop(img_list):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('D:/DATA SET FOR DARK SKINE/PPB/Code/SUP/shape_predictor_68_face_landmarks.dat')
    crp=[]
    for image in img_list:
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 1)
        for (i, rect) in enumerate(rects):
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            (x, y, w, h) = face_utils.rect_to_bb(rect)
        crop_img = gray[y:y+h, x:x+w]
        crp.append(crop_img)
    return crp




# GENERATE LABEL
def lbl_gen(lbl_path):
    lbl=[]
    with open(lbl_path) as doc:
        label=csv.reader(doc,delimiter=',')
        for row in label:
            lbl.append(row[1])
    
    return lbl



# Creat Numaric Label
def b_lbl(y_train,lbl_ls):
    lbl={}
    for lb in range(len(lbl_ls)):
        lbl[lbl_ls[lb]]=lb
    lbl_b=[]
    for lb in y_train:
        lbl_b.append(lbl[lb])
    return lbl_b




# GENERATE FILE NAME
def cd(fln):
    nm=len(str(fln))
    if nm==1:
        return "000"+str(fln)
    elif nm==2:
        return "00"+str(fln)
    elif nm==3:
        return "0"+str(fln)
    else:
        return str(fln)
    

# FACE COORDINATION
def rect_to_bb(rect):
	# take a bounding predicted by dlib and convert it
	# to the format (x, y, w, h) as we would normally do
	# with OpenCV
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
 
	# return a tuple of (x, y, w, h)
	return (x, y, w, h)

'''
def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)
 
	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
 
	# return the list of (x, y)-coordinates
	return coords
'''

# Convert to Matrix
def matrix(st):
    numImages=len(st)
    sz=st[0].shape
    data=np.zeros((numImages,sz[0]*sz[1]), dtype=np.uint8)
    for i in range(numImages):
        image=st[i].flatten()
        data[i,:]=image
    return data




# SELECT LAND MARKS (LANDMARK POINT )
def Bjet(img,fimg):
    js=[]
    def jt(img,x,y,m):
        return img[y-m:y+m,x-m:x+m]
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('D:/DATA SET FOR DARK SKINE/PPB/Code/FR_PPB/Classifier/dlib/shape_predictor_68_face_landmarks.dat')
    rects = detector(img, 1)
    for (i, rect) in enumerate(rects):
        shape = predictor(img, rect)
        shape = face_utils.shape_to_np(shape)
        p=[36,42,33,48,54,8]
        for i in p:
            (x,y)=shape[i]
            js.append(jt(fimg,x,y,5))
    return js




# CREAT GRAPH
def Bgf(img):
    ksize=50
    lamda=[4,4*np.sqrt(2),8,8*np.sqrt(2),16]
    sigma=lamda
    gamma=1
    phi=[0, np.pi/2]
    
    bg={'b0':[],
        'b1':[],
        'b2':[],
        'b3':[],
        'b4':[],
        'b5':[]}
    for i in range(1,8):
        theta=i*np.pi/8
        for lam in range(len(lamda)):
            for ph in range(len(phi)):
                kernel=cv2.getGaborKernel((ksize,ksize),sigma[lam],theta,lamda[lam],gamma,phi[ph],ktype=cv2.CV_32F)
                fimg=cv2.filter2D(img,cv2.CV_8UC3,kernel)
                tmp=Bjet(img,fimg)
                for j in range(len(tmp)):
                    bg[list(bg.keys())[j]].append(tmp[j])
    return bg