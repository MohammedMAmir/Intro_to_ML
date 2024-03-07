import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix 


def fit_NeuralNetwork(X_train,y_train,alpha,hidden_layer_sizes,epochs):
    #Enter implementation here
    
def forwardPropagation(x, weights):
    #Enter implementation here

def errorPerSample(X,y_n):
    #Enter implementation here

def backPropagation(X,y_n,s,weights):
    #Enter implementation here

def updateWeights(weights,g,alpha):
    #Enter implementation here

def activation(s):
    if s < 0:
        return 0
    else:
        return s

def derivativeActivation(s):
    #Enter implementation here

def outputf(s):
    #Enter implementation here

def derivativeOutput(s):
    #Enter implementation here

def errorf(x_L,y):
    #Enter implementation here

def derivativeError(x_L,y):
    #Enter implementation here

def pred(x_n,weights):
    #Enter implementation here
    
def confMatrix(X_train,y_train,w):
    N = X_train.shape[0]
    #Create a column of ones and append it to X_train
    ones = np.ones((N,1))
    modified_X_train = np.hstack((ones, X_train))

    #Initialize counts:
    true_neg = 0
    true_pos = 0
    false_neg = 0
    false_pos = 0

    #Go through all the samples
    for i in range(N):
        #If y_i is 1:
        if y_train[i] == 1:
            #If the perceptron correctly outputs 1, increase true_pos
            if(pred(modified_X_train[i], w) == y_train[i]):
                true_pos += 1
            #else, increase false_pos
            else:
                false_pos += 1
        #If y_i is -1:
        else:
            #If the perceptron correctly outputs -1, increase true_neg
            if(pred(modified_X_train[i], w) == y_train[i]):
                true_neg += 1
            #else, increase false_neg
            else:
                false_neg += 1
    
    #Create a two by two confusion matrix and fill it with respective counts
    result = np.arange(4).reshape(2,2)
    result[0][0] = true_neg
    result[0][1] = false_pos
    result[1][0] = false_neg
    result[1][1] = true_pos
    return result

def plotErr(e,epochs):
    #Enter implementation here
    stub = 0
    return stub
    
def test_SciKit(X_train, X_test, Y_train, Y_test):
    pct = MLPClassifier(solver="adam", alpha=1e-5, hidden_layer_sizes=(30,10), random_state=1)
    pct.fit(X_train, Y_train)
    confusion_matrix(Y_test, pct.predict(X_test))


def test_Part1():
    from sklearn.datasets import load_iris
    X_train, y_train = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X_train[50:],y_train[50:],test_size=0.2, random_state=1)
    
    for i in range(80):
        if y_train[i]==1:
            y_train[i]=-1
        else:
            y_train[i]=1
    for j in range(20):
        if y_test[j]==1:
            y_test[j]=-1
        else:
            y_test[j]=1
        
    err,w=fit_NeuralNetwork(X_train,y_train,1e-2,[30, 10],100)
    
    plotErr(err,100)
    
    cM=confMatrix(X_test,y_test,w)
    
    sciKit=test_SciKit(X_train, X_test, y_train, y_test)
    
    print("Confusion Matrix is from Part 1a is: ",cM)
    print("Confusion Matrix from Part 1b is:",sciKit)

test_Part1()
