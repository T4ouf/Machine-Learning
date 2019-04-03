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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import time

#Set numpy to fully print arrays (for confusion matrix)
np.set_printoptions(threshold=np.inf)

""" Question 3 K Neighbors - 2 neighbors -

# We load datas
training = np.load('../data/trn_img.npy')
trainingLabel = np.load('../data/trn_lbl.npy')

dev = np.load('../data/dev_img.npy')
devLabel = np.load('../data/dev_lbl.npy')

test = np.load('../data/tst_img.npy')

#Start the chrono
start_time = time.time()

#We use the learning algorithm
clf2 = KNeighborsClassifier(n_neighbors = 2)
clf2.fit(training, trainingLabel)

#compute learning time
learn_time = time.time()-start_time
print('\n\nKNeighbors Learning time : ' + str(format(learn_time,'.4f')) + ' sec')

#We try to predict on dev images in order to compute the error rate
pred_Label_Neighbors = clf2.predict(dev)
errorDev2 = dev[devLabel != pred_Label_Neighbors]
error2 = (errorDev2.shape[0]*1.0/devLabel.shape[0])*100

print('Neighbors Classification Error : ' + str(error2) + "%")

#compute predict time
start_time = time.time()
pred_KNeighbors_test = clf2.predict(test)
test_time = time.time() - start_time
print('KNeighbors Testing time : ' + str(format(test_time,'.4f')) + ' sec\n')


#We look for the confusion matrix
Neighbors_Confusion_Matrix = confusion_matrix(devLabel, pred_Label_Neighbors)

print('Neighbors confusion matrix :\n')
print(Neighbors_Confusion_Matrix)

"""

#Function that automatize the KNeighbors algorithm and prediction
def KNeighbors(training,trainingLabel,dev,devLabel,test, neighbors) :

    #Set numpy to fully print arrays
    np.set_printoptions(threshold=np.inf)

    #Start the chrono
    start_time = time.time()

    #We use the learning algorithm
    clf2 = KNeighborsClassifier(n_neighbors = neighbors)
    clf2.fit(training, trainingLabel)

    #compute learning time
    learn_time = time.time()-start_time
    print('\tKNeighbors Learning time : ' + str(format(learn_time,'.4f')) + ' sec')

    #We try to predict on dev images in order to compute the error rate
    pred_Label_Neighbors = clf2.predict(dev)
    errorDev2 = dev[devLabel != pred_Label_Neighbors]
    error2 = (errorDev2.shape[0]*1.0/devLabel.shape[0])*100
    print('\tNeighbors Classification Error : ' + str(error2) + "%")

    #compute predict time
    start_time = time.time()
    pred_KNeighbors_test = clf2.predict(test)
    test_time = time.time() - start_time
    print('\tKNeighbors Testing time : ' + str(format(test_time,'.4f')) + ' sec\n')

    return error2

###############################################################################

""" KNeighbors : varying number of Neighbors

# We load datas
training = np.load('../data/trn_img.npy')
trainingLabel = np.load('../data/trn_lbl.npy')

dev = np.load('../data/dev_img.npy')
devLabel = np.load('../data/dev_lbl.npy')

test = np.load('../data/tst_img.npy')

#prepare our arrays for the data
neighborsNumber = np.zeros(8)
errors = np.zeros(8)

j = 0

#make the number of neighbors varying
for i in range(2,10,1) :
    neighborsNumber[j]=i
    print('Neighbors = ' + str(i))
    errors[j] = KNeighbors(training,trainingLabel,dev,devLabel,test, i)
    j=j+1

#We prepare the plot
axes = plt.gca()
axes.set_xlim([0,10])
axes.set_ylim([0,30])

plt.figure(1)
plt.plot(neighborsNumber,errors,'b-',label='KNeighbors classifier')
plt.legend()
plt.title("Errors depending on the number of neighbors")       #Adding a title
plt.xlabel("n neighbors")         #X axis title
plt.ylabel("Error rate (%)")    #Y axis title
plt.legend()
plt.show()

t = input("Press Enter to finish the programm...")

"""


###############################################################################

""" KNeighbors (5 neighbors) + PCA """

# We load datas
training = np.load('../data/trn_img.npy')
trainingLabel = np.load('../data/trn_lbl.npy')

dev = np.load('../data/dev_img.npy')
devLabel = np.load('../data/dev_lbl.npy')

test = np.load('../data/tst_img.npy')

#prepare our arrays for the data
dimensions = np.zeros(20)
errors = np.zeros(20)

#index for the arrays (up)
j=0

#Apply PCA with different dimensions
for i in range(5,105,5):

    #Applying PCA
    pca = PCA(n_components = i)

    #fit our images to the new dimensions
    PCAtraining = pca.fit_transform(training)
    PCAdev = pca.transform(dev)
    PCAtest= pca.transform(test)

    dimensions[j]=i

    #Compute error and timings
    print('Dim : ' + str(i))
    errors[j]=KNeighbors(PCAtraining,trainingLabel,PCAdev,devLabel,PCAtest, 5)

    j=j+1

#Compute KNeighbors without PCA
print('Dim : 784 ')
errorWithoutPCA = KNeighbors(training,trainingLabel,dev,devLabel,test, 5)
x = [0,100]
y = [errorWithoutPCA, errorWithoutPCA]


#We prepare the plot
axes = plt.gca()
axes.set_xlim([0,110])
axes.set_ylim([0,40])

#We plot
plt.figure(1)
plt.plot(dimensions,errors,'b-',label='KNeighbors classifier With PCA')
plt.legend()
plt.title("Errors depending on the dimension kept by PCA")       #Adding a title
plt.xlabel("Dimension")         #X axis title
plt.ylabel("Error rate (%)")    #Y axis title

plt.figure(1)
plt.plot(x,y,'r-', label='KNeighbors classifier Without PCA')
plt.legend()
plt.show()

t = input("Press Enter to finish the programm...")
