# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import affichage


training = np.load('../data/trn_img.npy')
trainingLabel = np.load('../data/trn_lbl.npy')

dev = np.load('../data/dev_img.npy')
devLabel = np.load('../data/dev_lbl.npy')

def DistMin(PCAtraining, PCAdev) :
    # Defining our classes
    class0 = PCAtraining[trainingLabel == 0]
    class1 = PCAtraining[trainingLabel == 1]
    class2 = PCAtraining[trainingLabel == 2]
    class3 = PCAtraining[trainingLabel == 3]
    class4 = PCAtraining[trainingLabel == 4]
    class5 = PCAtraining[trainingLabel == 5]
    class6 = PCAtraining[trainingLabel == 6]
    class7 = PCAtraining[trainingLabel == 7]
    class8 = PCAtraining[trainingLabel == 8]
    class9 = PCAtraining[trainingLabel == 9]

    # Centroids of classes
    avg0 = np.mean(class0, axis=0)
    avg1 = np.mean(class1, axis=0)
    avg2 = np.mean(class2, axis=0)
    avg3 = np.mean(class3, axis=0)
    avg4 = np.mean(class4, axis=0)
    avg5 = np.mean(class5, axis=0)
    avg6 = np.mean(class6, axis=0)
    avg7 = np.mean(class7, axis=0)
    avg8 = np.mean(class8, axis=0)
    avg9 = np.mean(class9, axis=0)

    avg = [avg0,avg1,avg2,avg3,avg4,avg5,avg6,avg7,avg8,avg9]


    classifierDevLabel = np.zeros(devLabel.shape)

    for i in range(0,devLabel.shape[0],1):

        squareDist = np.zeros(10)

        for classNb in range(0,10,1):
            squareDist[classNb] = np.sum(((PCAdev[i] - avg[classNb])**2))

        classifierDevLabel[i] = np.argmin(squareDist)


    errorDev = PCAdev[devLabel != classifierDevLabel]
    error = (errorDev.shape[0]*1.0/devLabel.shape[0])*100
    return error



dimensions = np.zeros(11)
errors = np.zeros(11)

for dim in range(0,30,3) :
    pca = PCA(n_components = dim)

    PCAtraining = pca.fit_transform(training)

    PCAdev = pca.transform(dev)

    dimensions[dim/3]=dim
    errors[dim/3] = DistMin(PCAtraining,PCAdev)
    print('Dim : ' + str(dim) + '\n\tClassification Error : ' + str(errors[dim/3]) + "%\n")


dimensions[10] = 784
errors[10] = DistMin(training,label)

affichage.printGraph(dimensions,errors,"Errors depending of the dimension kept by PCA","Dimension","Error rate (%)")

t = input()
