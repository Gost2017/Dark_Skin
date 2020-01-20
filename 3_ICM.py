# -*- coding: utf-8 -*-
"""
Created on Tue Dec 24 23:52:15 2019

@author: Admin
"""

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score,classification_report
from sklearn.decomposition import FastICA
from sklearn.svm import SVC
import numpy as np
import pandas as pd
import seaborn as sn

n_components = 900

ica = FastICA(n_components=n_components,
              random_state=0,
              whiten=True).fit(data_set['Xtrain_gray'])

X_train_ica = ica.transform(data_set['Xtrain_gray'])
X_test_ica = ica.transform(data_set['Xtest_gray'])

ica_c = ica.components_.reshape((n_components, 36, 36))

plt.imshow(ica_c[300],cmap='gray')
plt.show


# Train a SVM classification model
clf=SVC(kernel='linear')
clf.fit(X_train_ica, data_set['ytrain'])

Y_pred = clf.predict(X_test_ica)


#Evaluation

classification_report(data_set['ytest'], Y_pred, target_names=label)

cf=confusion_matrix(data_set['ytest'], Y_pred, labels=label)   # Conf Matrix
cm_normalized = cf.astype('float') / cf.sum(axis=1)[:, np.newaxis]        #NORMALIZE Conf Matrix

accuracy_score(data_set['ytest'], Y_pred, normalize=False)
df_cm = pd.DataFrame(cm_normalized, index = [i for i in list(label)],
                  columns = [i for i in list(label)])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)
