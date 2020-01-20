# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 23:50:16 2019

@author: Admin
"""
import cv2
from support import load, re_size, crop, matrix, gray
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import numpy as np
import random

#rawdata=[]
#lbl=[]

# LOAD RAW DATA
# RESIZE ALL IMAGE TO 128,182
# CONVERT IT TO GRAY

data=load('Label/label.csv')

modify=[]
for i in range(len(data['raw_img'])):
    if data['raw_img'][i].shape<(250,255):
        modify.append(cv2.pyrUp(data['raw_img'][i]))
    else:
        modify.append(data['raw_img'][i])
modify2=[]
for i in range(len(modify)):
    if modify[i].shape<(500,333):
        modify2.append(cv2.pyrUp(modify[i]))
    else:
        modify2.append(modify[i])

data['re_size']=re_size(modify2,333,500)

data['gray']=gray(data['re_size'])

del modify2,modify,i

'''
b=set()
for i in range(len(modify2)):
    b.add(modify2[i].shape)
del i


st={}
for i in b:
    st[i]=0
for i in range(len(modify2)):
    st[modify2[i].shape]+=1

'''

# CROP IMAGE TO ONLY FACE IMAGE
import dlib
from imutils import face_utils

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('D:/DATA SET FOR DARK SKINE/PPB/Code/FR_PPB/Classifier/dlib/shape_predictor_68_face_landmarks.dat')
crp=[]
for gry in data['re_size']:
    rects = detector(gry, 1)
    for (i, rect) in enumerate(rects):
        shape = predictor(gry, rect)
        shape = face_utils.shape_to_np(shape)
        (x, y, w, h) = face_utils.rect_to_bb(rect)
    crop_img = gry[y:y+h, x:x+w]
    crp.append(crop_img)

data['crop_img']=re_size(crp,36,36)

del crp, i, rect, detector, predictor,x,y,w,h,shape,gry,crop_img
'''
data.keys()
i=1180
plt.imshow(data['crop_img'][i],cmap='gray')
plt.show()

plt.imshow(data['re_size'][i],cmap='gray')
plt.show()
del i
'''
'''
b=set()
for i in range(len(crp)):
    b.add(crp[i].shape)
del i


st={}
for i in b:
    st[i]=0    
for i in range(len(crp)):
    st[crp[i].shape]=str(st[crp[i].shape]) + ',' + str(i)
'''

# RANDOM DISTRIBUTION TRAIN AND TEST SET

'DEFINE TRAIN SET AND TEST SET 80% vs 20%'
train_set={'F_TY_I':60,'F_TY_II':119,'F_TY_III':49,'F_TY_IV':19,'F_TY_V':116,'F_TY_VI':88,
           'M_TY_I':77,'M_TY_II':135,'M_TY_III':51,'M_TY_IV':18,'M_TY_V':71,'M_TY_VI':162}

test_set={'F_TY_I':15,'F_TY_II':29,'F_TY_III':12,'F_TY_IV':4,'F_TY_V':28,'F_TY_VI':21,
           'M_TY_I':19,'M_TY_II':33,'M_TY_III':12,'M_TY_IV':4,'M_TY_V':17,'M_TY_VI':40}

pol=random.sample(range(1199),1199)

x_train=[]
x_train_c=[]
y_train=[]

x_test=[]
x_test_c=[]
y_test=[]


dn_set=[]
'Make Train Set'
for i in pol:
    if train_set[data['label'][i]]>0:
        x_train.append(data['gray'][i])
        x_train_c.append(data['crop_img'][i])
        y_train.append(data['label'][i])
        dn_set.append(i)
        train_set[data['label'][i]]-=1
'remove train value from pol'
for i in dn_set:
    del pol[pol.index(i)]
del i

'Make Test Set'
for i in pol:
    if test_set[data['label'][i]]>0:
        x_test.append(data['gray'][i])
        x_test_c.append(data['crop_img'][i])
        y_test.append(data['label'][i])
        dn_set.append(i)
        test_set[data['label'][i]]-=1

# GROUP TO SOURCE DICTIONARY

'Grouping'
source={'x_train':x_train,
        'y_train':y_train,
        'x_train_c':x_train_c,
        'x_test':x_test,
        'y_test':y_test,
        'x_test_c':x_test_c}

del pol,dn_set,test_set,train_set,x_train,y_train,x_test,y_test,i, x_test_c,x_train_c
data.keys()
source.keys()

'''
# CONVERT TO GRAY SCALE 

source_g={}
for dt in ['x_train','x_test']:
    for im in source[dt]:
        im=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        source_g.setdefault(dt,[]).append(im)
'''

# CONVER IMAGE TO MATRIX

'Train & Test Matrix'
data_set={ 'Xtrain_gray':matrix(source['x_train']),
          'Xtest_gray':matrix(source['x_test']),
          'ytrain':source['y_train'],
          'ytest':source['y_test']}

label=['M_TY_I','M_TY_II','M_TY_III','M_TY_IV','M_TY_V','M_TY_VI',
      'F_TY_I','F_TY_II','F_TY_III','F_TY_IV','F_TY_V','F_TY_VI',]

print('data_set -->  Xtrain_gray , Xtest_gray , ytrain , ytest')
print('source -->  x_train, y_train, x_train_c, x_test, y_test, x_test_c')



