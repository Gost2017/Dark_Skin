# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 23:03:50 2019

@author: Admin
"""

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score,classification_report
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import numpy as np
import pandas as pd
import seaborn as sn

#EXTRACT PRINCIPAL COMPONANT

n_components = 600

pca = PCA(n_components=n_components,whiten=True).fit(data_set['Xtrain_gray'])


# NUMBER OF PROPER COMPONENT
from matplotlib import pyplot as plt
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');

'Eigenfaces_set'
eigenfaces = pca.components_.reshape((n_components, 500, 333))

X_train_pca = pca.transform(data_set['Xtrain_gray'])
X_test_pca = pca.transform(data_set['Xtest_gray'])


# Train a SVM classification model
clf=SVC(kernel='linear',C=0.1)
clf.fit(X_train_pca, data_set['ytrain'])

Y_pred = clf.predict(X_test_pca)


#Evaluation

classification_report(data_set['ytest'], Y_pred, target_names=label)

cf=confusion_matrix(data_set['ytest'], Y_pred, labels=label)   # Conf Matrix
cm_normalized = cf.astype('float') / cf.sum(axis=1)[:, np.newaxis]        #NORMALIZE Conf Matrix

accuracy_score(data_set['ytest'], Y_pred, normalize=False)
df_cm = pd.DataFrame(cm_normalized, index = [i for i in list(label)],
                  columns = [i for i in list(label)])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)


# Plot_Eigenfaces
import random
s=random.sample(range(0,len(eigenfaces)),20)
for i in range(len(s)):
    img = eigenfaces[i]
    plt.subplot(4,5,1+i)
    plt.imshow(img, cmap='gray')
    plt.tick_params(labelleft='off', labelbottom='off', bottom='off',top='off',right='off',left='off', which='both')
plt.show()