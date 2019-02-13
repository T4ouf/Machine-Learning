# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt


training = np.load('../data/trn_img.npy')
trainingLabel = np.load('../data/trn_lbl.npy')

dev = np.load('../data/dev_img.npy')
devLabel = np.load('../data/dev_lbl.npy')


"""
Function that prints an image (reshape is in the function)
"""
def printImage(im):
    img = im.reshape(28,28)
    plt.imshow(img,plt.cm.gray)
    plt.show()


# Defining our classes
class0 = training[trainingLabel == 0]
class1 = training[trainingLabel == 1]
class2 = training[trainingLabel == 2]
class3 = training[trainingLabel == 3]
class4 = training[trainingLabel == 4]
class5 = training[trainingLabel == 5]
class6 = training[trainingLabel == 6]
class7 = training[trainingLabel == 7]
class8 = training[trainingLabel == 8]
class9 = training[trainingLabel == 9]

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
        squareDist[classNb] = np.sum(((dev[i] - avg[classNb])**2))
    
    classifierDevLabel[i] = np.argmin(squareDist)        


errorDev = dev[devLabel != classifierDevLabel]
error = (errorDev.shape[0]/devLabel.shape[0])*100
print('Error of classification : ' + str(error) + "%")