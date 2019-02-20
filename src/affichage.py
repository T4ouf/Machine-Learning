import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
plt.ion()

MAXITER=20

def affiche(W1, W2, A, iteration):
    cmap = cm.jet
    gradient=cmap(np.linspace(0.0,1.0,12))
    # Separe le poids du vecteur normal
    a0 = A[0]
    A12 = A[1:3]
    # Longueur du vecteur normal
    norm = np.sqrt((A12**2).sum())
    # Determine le point le plus proche de l'origine
    H = - a0 * A12 / (norm * norm)
    # vecteur directeur - rotation de 90deg par produit matriciel
    # U = np.array([[0, 1], [-1, 0]]) @ A12 / norm
    U = np.dot( np.array([[0, 1], [-1, 0]]) , A12) / norm
    # segment autour de H et de longueur 10
    D = np.vstack((H + 5*U, H - 5*U))
    # pour l'affichage des points
    plt.axis([-5, 5, -5, 5])
    plt.plot(W1[0,:], W1[1,:], 'r+')
    plt.plot(W2[0,:], W2[1,:], 'bx')
    plt.plot(D[:,0], D[:,1], ls='-', color=gradient[iteration%(len(gradient))])
    # plt.show()


"""
Function that prints an image (reshape is in the function)
"""
def printImage(im):
    img = im.reshape(28,28)
    plt.imshow(img,plt.cm.gray)
    plt.show()

def printGraph(x,y,title,xAxis,yAxis) :
    plt.plot(x,y,'b-')

    plt.title(title)  #Adding a title
    plt.xlabel(xAxis) #X axis title
    plt.ylabel(yAxis) #Y axis title
    plt.show()
