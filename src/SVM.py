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
from sklearn.metrics import confusion_matrix
from sklearn import svm
import time

#Set numpy to fully print arrays
np.set_printoptions(threshold=np.inf)

""" Question 3 Support Vector Machine 

start_time = time.time()

# We load datas
training = np.load('../data/trn_img.npy')
trainingLabel = np.load('../data/trn_lbl.npy')

dev = np.load('../data/dev_img.npy')
devLabel = np.load('../data/dev_lbl.npy')

test = np.load('../data/tst_img.npy')

clf = svm.SVC(gamma = 'scale')
clf.fit(training, trainingLabel)

#Compute the learning time
learn_time = time.time()-start_time
print('SVM Learning time : ' + str(format(learn_time,'.4f')) + ' sec')

#We predict the dev labels (to compute error rate)
pred_Label_SVM = clf.predict(dev)

#We compute the error rate
errorDev = dev[devLabel != pred_Label_SVM]
error = (errorDev.shape[0]*1.0/devLabel.shape[0])*100

print('SVM Classification Error : ' + str(error) + "%")

#We compute the Classification time
start_time = time.time()
pred_SVM_test = clf.predict(test)
test_time = time.time() - start_time
print('SVM Testing time : ' + str(format(test_time,'.4f')) + ' sec')

#We compute and print the Confusion Matrix
SVM_Confusion_Matrix = confusion_matrix(devLabel, pred_Label_SVM)

print('SVM confusion matrix :\n')
print(SVM_Confusion_Matrix)

"""

###############################################################################

def PCA_SVM(training,trainingLabel,dev,devLabel,test):

    start_time = time.time()

    #Use the learning algorithm
    clf = svm.SVC(gamma='scale')
    clf.fit(training, trainingLabel)

    #Compute the learning time
    learn_time = time.time()-start_time
    print('\tSVM Learning time : ' + str(format(learn_time,'.4f')) + ' sec')

    #We predict the dev labels (to compute error rate)
    pred_Label_SVM = clf.predict(dev)

    #We compute the error rate
    errorDev = dev[devLabel != pred_Label_SVM]
    error = (errorDev.shape[0]*1.0/devLabel.shape[0])*100

    print('\tSVM Classification Error : ' + str(error) + "%")

    #We compute the Classification time
    start_time = time.time()
    pred_SVM_test = clf.predict(test)
    test_time = time.time() - start_time
    print('\tSVM Testing time : ' + str(format(test_time,'.4f')) + ' sec')

    return error

""" PCA + SVM """

start_time = time.time()

# We load datas
training = np.load('../data/trn_img.npy')
trainingLabel = np.load('../data/trn_lbl.npy')

dev = np.load('../data/dev_img.npy')
devLabel = np.load('../data/dev_lbl.npy')

test = np.load('../data/tst_img.npy')

#prepare our arrays for the data
dimensions = np.zeros(27)
errors = np.zeros(27)

j = 0

#Using PCA :
for dim in range(5,105,5) :
    pca = PCA(n_components = dim)

    #fit our datas to new dimensions
    PCAtraining = pca.fit_transform(training)
    PCAdev = pca.transform(dev)
    PCAtest= pca.transform(test)

    print('Dim : ' + str(dim))

    dimensions[j]=dim
    errors[j]=PCA_SVM(PCAtraining,trainingLabel,PCAdev,devLabel,PCAtest)

    j=j+1

#Using PCA :
for dim in range(200,800,100) :
    pca = PCA(n_components = dim)

    #fit our datas to new dimensions
    PCAtraining = pca.fit_transform(training)
    PCAdev = pca.transform(dev)
    PCAtest= pca.transform(test)

    print('Dim : ' + str(dim))

    dimensions[j]=dim
    errors[j]=PCA_SVM(PCAtraining,trainingLabel,PCAdev,devLabel,PCAtest)

    j=j+1

#Compute SVM without PCA
print('Dim : 784')
errorWithoutPCA = PCA_SVM(training,trainingLabel,dev,devLabel,test)

dimensions[j]=784
errors[j]=errorWithoutPCA

x = [0,800]
y = [errorWithoutPCA, errorWithoutPCA]


#We prepare the plot
axes = plt.gca()
axes.set_xlim([0,800])
axes.set_ylim([0,30])

plt.figure(1)
plt.plot(dimensions,errors,'b-',label='SVM classifier With PCA')
plt.legend()
plt.title("Errors depending on the dimension kept by PCA")       #Adding a title
plt.xlabel("Dimension")         #X axis title
plt.ylabel("Error rate (%)")    #Y axis title

plt.figure(1)
plt.plot(x,y,'r-', label='SVM classifier Without PCA')
plt.legend()
plt.show()


t = input("Press Enter to finish the programm...")



###############################################################################

""" Optimal Classification System

# We load datas
training = np.load('../data/trn_img.npy')
trainingLabel = np.load('../data/trn_lbl.npy')

dev = np.load('../data/dev_img.npy')
devLabel = np.load('../data/dev_lbl.npy')

test = np.load('../data/tst_img.npy')

#fix the PCA to 500
pca = PCA(n_components = 500)

#fit our datas to the PCA
PCAtraining = pca.fit_transform(training)
PCAtest= pca.transform(test)

#Using SVM
clf = svm.SVC(gamma='scale')
clf.fit(PCAtraining, trainingLabel)

#Predict on our test images
pred_SVM_test = clf.predict(PCAtest)

#code to save the test results
np.save("test.npy", pred_SVM_test)
"""
