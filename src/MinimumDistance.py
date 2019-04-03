# -*- coding: utf-8 -*-

#################################################################################
#                     MACHINE LEARNING - IMAGE RECOGNITION PROJECT              #
#                           POLYTECH ET4 / 2018 - 2019                          #
#                                                                               #
#                      Eurydice Ruggieri - Thomas von Ascheberg                 #
#################################################################################

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import time

""" Question 1 : Minimum Distance classifier 

# We load datas
training = np.load('../data/trn_img.npy')
trainingLabel = np.load('../data/trn_lbl.npy')

dev = np.load('../data/dev_img.npy')
devLabel = np.load('../data/dev_lbl.npy')

test = np.load('../data/tst_img.npy')


start_time = time.time()

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

#Compute the learning time
learn_time = time.time()-start_time

#We try to guess the classes of our dev images
classifierDevLabel = np.zeros(devLabel.shape)

#For Each image...
for i in range(0,devLabel.shape[0],1):

    squareDist = np.zeros(10)

    #Computing the distance to each centroid
    for classNb in range(0,10,1):
        squareDist[classNb] = np.sum(((dev[i] - avg[classNb])**2))

    #We say that the image's class is the class that is the nearest (minimum distance classifier)
    classifierDevLabel[i] = np.argmin(squareDist)

#Checking the error rate on dev
errorDev = dev[devLabel != classifierDevLabel]
error = (errorDev.shape[0]*1.0/devLabel.shape[0])*100
print('Classification Error : ' + str(error) + "%")
print('Learning time : ' + str(format(learn_time,'.4f')) + ' sec')


start_time = time.time()

#Generating test files
#We try to guess the classes of our dev images
classifierTestLabel = np.zeros(test.shape[0])

#For Each image...
for i in range(0,test.shape[0],1):

    squareDist = np.zeros(10)

    #Computing the distance to each centroid
    for classNb in range(0,10,1):
        squareDist[classNb] = np.sum(((test[i] - avg[classNb])**2))

    #We say that the image's class is the class that is the nearest (minimum distance classifier)
    classifierTestLabel[i] = np.argmin(squareDist)

test_time = time.time() - start_time
print('Testing time : ' + str(format(test_time,'.4f')) + ' sec')

#code to save the test results
#np.save("testMinDist.npy", classifierTestLabel)

"""

###############################################################################

""" Q2 : Minimum Distance Classifier + PCA """

# We load datas
training = np.load('../data/trn_img.npy')
trainingLabel = np.load('../data/trn_lbl.npy')

dev = np.load('../data/dev_img.npy')
devLabel = np.load('../data/dev_lbl.npy')

test = np.load('../data/tst_img.npy')

"""
Function that mimic a Minimum distance classifier
(used in the DistMin.py file)
"""
def DistMin(PCAtraining, PCAdev) :

    #defining our start time
    start_time = time.time()

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

    #Compute the learning time
    learn_time = time.time()-start_time
    print('\tLearning time : ' + str(learn_time) + ' sec')


    start_time = time.time()

    #We try to guess the classes of our dev images
    classifierDevLabel = np.zeros(devLabel.shape)

    #For Each image...
    for i in range(0,devLabel.shape[0],1):

        squareDist = np.zeros(10)

        #Computing the distance to each centroid
        for classNb in range(0,10,1):
            squareDist[classNb] = np.sum(((PCAdev[i] - avg[classNb])**2))

        #We say that the image's class is  the class that is the nearest (minimum distance classifier)
        classifierDevLabel[i] = np.argmin(squareDist)

    test_time = time.time() - start_time
    print('\tTesting time : ' + str(test_time) + ' sec')


    #Checking the error rate on dev
    errorDev = PCAdev[devLabel != classifierDevLabel]
    error = (errorDev.shape[0]*1.0/devLabel.shape[0])*100
    return error


dimensions = np.zeros(10)
errors = np.zeros(10)

#Using PCA :
#Testing Min Dist Classifier on Compressed image (reduced dimensions from initial 784 to dim (from 0 to 30))
for dim in range(0,30,3) :
    pca = PCA(n_components = dim)

    PCAtraining = pca.fit_transform(training)

    PCAdev = pca.transform(dev)

    i = dim//3

    dimensions[i]=dim
    print('Dim : ' + str(dim))
    errors[i] = DistMin(PCAtraining,PCAdev)
    print('\tClassification Error : ' + str(errors[i]) + "%")


print('Dim : 784')
errorWithoutPCA = DistMin(training, dev)
print('\tClassification Error : ' + str(errorWithoutPCA) + "%")
errorWithoutPCAarray = np.arange(0, 10, dtype=np.float)
erroerrorWithoutPCAarray = errorWithoutPCA

#dimensions[10] = 784
#errors[10] = DistMin(training,dev)

x = [0,30]
y = [errorWithoutPCA, errorWithoutPCA]

axes = plt.gca()
axes.set_xlim([0,30])
axes.set_ylim([0,100])

plt.figure(1)
plt.plot(dimensions,errors,'b-',label='Minimum distance classifier With PCA')
plt.legend()
plt.title("Errors depending on the dimension kept by PCA")       #Adding a title
plt.xlabel("Dimension") #X axis title
plt.ylabel("Error rate (%)") #Y axis title

plt.figure(1)
plt.plot(x,y,'r-', label='Minimum distance classifier Without PCA')
plt.legend()
plt.show()


t = input("Press Enter to finish the programm...")
