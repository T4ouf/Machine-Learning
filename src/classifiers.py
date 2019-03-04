# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn import svm
import affichage

#Fully prints the numpy arrays
np.set_printoptions(threshold=np.inf)


# We load datas
training = np.load('../data/trn_img.npy')
trainingLabel = np.load('../data/trn_lbl.npy')

dev = np.load('../data/dev_img.npy')
devLabel = np.load('../data/dev_lbl.npy')

clf = svm.SVC(kernel='linear')
clf.fit(training, trainingLabel) 

pred_Label_SVM = clf.predict(dev)

errorDev = dev[devLabel != pred_Label_SVM]
error = (errorDev.shape[0]*1.0/devLabel.shape[0])*100

print('\nSVM Classification Error : ' + str(error) + "%\n")

SVM_Confusion_Matrix = confusion_matrix(devLabel, pred_Label_SVM)

print(SVM_Confusion_Matrix)

clf2 = KNeighborsClassifier(n_neighbors = 2)
clf2.fit(training, trainingLabel)

pred_Label_Neighbors = clf2.predict(dev)
errorDev2 = dev[devLabel != pred_Label_Neighbors]
error2 = (errorDev2.shape[0]*1.0/devLabel.shape[0])*100

#error = SVM_PCA(training, dev, 30)
print('\nNeighbors Classification Error : ' + str(error2) + "%\n")

Neighbors_Confusion_Matrix = confusion_matrix(devLabel, pred_Label_Neighbors)

print(Neighbors_Confusion_Matrix)