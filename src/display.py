#################################################################################
#                     MACHINE LEARNING - IMAGE RECOGNITION PROJECT              #
#                           POLYTECH ET4 / 2018 - 2019                          #
#                                                                               #
#                      Eurydice Ruggieri - Thomas von Ascheberg                 #
#################################################################################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
plt.ion()

"""
Function that prints an image (reshape is in the function)
"""
def printImage(im):
    img = im.reshape(28,28)
    plt.imshow(img,plt.cm.gray)
    plt.show()

"""
Function that prints a graph
"""
def printGraph(x,y,title,xAxisLabel,yAxisLabel,xmin,xmax,ymin,ymax,figure) :
    
    axes = plt.gca()
    
    axes.set_xlim([xmin,xmax])
    axes.set_ylim([ymin,ymax])
    
    plt.figure(figure)
    plt.plot(x,y,'b-')
    plt.title(title)       #Adding a title
    plt.xlabel(xAxisLabel) #X axis title
    plt.ylabel(yAxisLabel) #Y axis title
   
