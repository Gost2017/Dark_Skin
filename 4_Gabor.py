# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 01:09:07 2019

@author: Admin
"""
import cv2
import matplotlib.pyplot as plt
import dlib
from imutils import face_utils
import numpy as np
from support import Bgf,Bjet,re_size
import datetime

pol=np.concatenate((data_set['Xtrain_gray'],data_set['Xtest_gray']),axis=0)


# HBGM TRAIN SET
mtx=data_set['Xtrain_gray']
mtx_b=mtx[84:,]
err=[] # RECORD WITH ERROR
bgph=[]
for i in range(len(mtx_b)):
    try:
        bgph.append(Bgf(mtx_b[i].reshape(500,333)))
    except:
        err.append(i)
        pass
'--------------------------------------------------------------'
    
Bgf(mtx[85].reshape(500,333))



for i in range(len(mtx)):
    bg=Bgf(mtx[i].reshape(500,333))
    nc=(bg['b0'][0]).shape[0]*(bg['b0'][0]).shape[1]
    rw=len(bg)*len(bg['b0'])
    rp=np.zeros((rw,nc),dtype=np.uint8)
    r=0
    for b in bg:
        for j in range(len(bg[b])):
            rp[r,:]=bg[b][0].flatten()
            r+=1
    if i==0:
        fim=np.zeros((len(mtx),rp.shape[0]*rp.shape[1]),dtype=np.uint8)
    fim[i,:]=rp.reshape(rp.shape[0]*rp.shape[1])
del b,bg,i,j,nc,r,rp,rw, mtx
print(datetime.datetime.now())


bg=Bgf(mtx[131].reshape(500,333))
nc=(bg['b0'][0]).shape[0]*(bg['b0'][0]).shape[1]

np.sum(fim[233],axis=0)
data_set['Xtest_gray'].shape
# cv2.circle(img, (x, y), 1, (0, 0, 255), -1)
# EXTRACT JET AT EACH LANGMARK
# IMAGE TETST

img=cv2.resize(data['gray'][238],(333,500))
img=img.reshape(-1)
data_set['Xtrain_gray'][85]=img

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

plt.imshow(mtx[84].reshape(500,333),cmap='gray')
plt.show()