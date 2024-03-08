import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix 


def fit_NeuralNetwork(X_train,y_train,alpha,hidden_layer_sizes,epochs):
    # Initialize the epoch errors
    err=np.zeros((epochs,1))
    
    # Initialize the architecture
    N, d = X_train.shape
    X0 = np.ones((N,1))
    X_train = np.hstack((X0,X_train))
    d=d+1
    L = len(hidden_layer_sizes)
    L=L+2
    
    #Initializing the weights for input layer
    weight_layer = np.random.normal(0, 0.1, (d,hidden_layer_sizes[0])) #np.ones((d,hidden_layer_sizes[0]))
    weights = []
    weights.append(weight_layer) #append(0.1*weight_layer)
    
    #Initializing the weights for hidden layers
    for l in range(L-3):
        weight_layer = np.random.normal(0, 0.1, (hidden_layer_sizes[l]+1,hidden_layer_sizes[l+1])) 
        weights.append(weight_layer) 

    #Initializing the weights for output layers
    weight_layer= np.random.normal(0, 0.1, (hidden_layer_sizes[l+1]+1,1)) 
    weights.append(weight_layer) 
    
    for e in range(epochs):
        choiceArray=np.arange(0, N)
        np.random.shuffle(choiceArray)
        errN=0
        for n in range(N):
            index=choiceArray[n]
            x=np.transpose(X_train[index])
            x_ret, s = forwardPropagation(x, weights)
            g = backPropagation(x_ret, y_train[index], s, weights)
            weights = updateWeights(weights, g, alpha)

        err[e]=errN/N 
    return err, weights
    
def forwardPropagation(x, weights):
    l=len(weights)+1
    currX = x
    retS=[]
    retX=[]
    retX.append(currX)

    # Forward Propagate for each layer
    for i in range(l-1):
        
        currS= np.dot(currX, weights[i])
        retS.append(currS)
        currX=currS
        if i != len(weights)-1:
            for j in range(len(currS)):
                currX[j]= activation(currS[j])
            currX= np.hstack((1,currX))
        else:
            currX= outputf(currS)
        retX.append(currX)
    return retX,retS

def errorPerSample(X,y_n):
    l = len(X)
    return errorf(X[l-1], y_n)

def backPropagation(X,y_n,s,weights):
    #x:0,1,...,L
    #S:1,...,L
    #weights: 1,...,L
    l=len(X)
    delL=[]

    # To be able to complete this function, you need to understand this line below
    # In this line, we are computing the derivative of the Loss function w.r.t the 
    # output layer (without activation). This is dL/dS[l-2]
    # By chain rule, dL/dS[l-2] = dL/dy * dy/dS[l-2] . Now dL/dy is the derivative Error and 
    # dy/dS[l-2]  is the derivative output.
    delL.insert(0,derivativeError(X[l-1],y_n)*derivativeOutput(s[l-2]))
    curr=0
    
    # Now, let's calculate dL/dS[l-2], dL/dS[l-3],...
    for i in range(len(X)-2, 0, -1): #L-1,...,0
        delNextLayer=delL[curr]
        WeightsNextLayer=weights[i]
        sCurrLayer=s[i-1]
        
        #Init this to 0s vector
        delN=np.zeros((len(s[i-1]),1))

        #Now we calculate the gradient backward
        #Remember: dL/dS[i] = dL/dS[i+1] * W(which W???) * activation
        for j in range(len(s[i-1])): #number of nodes in layer i - 1
            for k in range(len(s[i])): #number of nodes in layer i
                #TODO: calculate delta at node j
                delN[j]=delN[j]+ WeightsNextLayer[j,k]*delNextLayer[k]*derivativeActivation(sCurrLayer[j])# Fill in the rest
        
        delL.insert(0,delN)
    
    # We have all the deltas we need. Now, we need to find dL/dW.
    # It's very simple now, dL/dW = dL/dS * dS/dW = dL/dS * X
    g=[]
    for i in range(len(delL)):
        rows,cols=weights[i].shape
        gL=np.zeros((rows,cols))
        currX=X[i]
        currdelL=delL[i]
        for j in range(rows):
            for k in range(cols):
                #TODO: Calculate the gradient using currX and currdelL
                gL[j,k]= np.dot(currX[j],currdelL[k])
        g.append(gL)
    return g

def updateWeights(weights,g,alpha):
    nW=[]
    for i in range(len(weights)):
        rows, cols = weights[i].shape
        currWeight=weights[i]
        currG=g[i]
        for j in range(rows):
            for k in range(cols):
                #TODO: Gradient Descent Update
                currWeight[j,k]= currWeight[j,k] + alpha*currG[j,k]
        nW.append(currWeight)
    return nW

def activation(s):
    if s < 0:
        return 0
    else:
        return s

def derivativeActivation(s):
    if s<0:
        return 0
    else:
        return 1
    
def outputf(s):
    return (1/(1+np.exp(-s)))

def derivativeOutput(s):
    return outputf(s)*(1-outputf(s))

def errorf(x_L,y):
    if y == 1:
        return np.log(x_L)
    if y == -1:
        return np.log(1-x_L)

def derivativeError(x_L,y):
    if y == 1:
        return -1/x_L
    if y == -1:
        return 1/(1-x_L)

def pred(x_n,weights):
    stub = 0
    return stub
    
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
    conf_mat = confusion_matrix(Y_test, pct.predict(X_test))
    return conf_mat


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
