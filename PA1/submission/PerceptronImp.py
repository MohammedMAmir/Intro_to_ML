import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import confusion_matrix 

#The output of this function is the vector w that represents the coefficients of the line computed by
#the pocket algorithm that best separates the two classes of training data points. The dimensions
#of this vector is d + 1 as the offset term is accounted in the computation.
def fit_perceptron(X_train, y_train):
    #The number of samples we have is stored in N
    N = X_train.shape[0]
    #The dimensionality of each data point is stored in d
    d = X_train.shape[1]
    #Create a column of ones and append it to X_train
    ones = np.ones((N,1))
    modified_X_train = np.hstack((ones, X_train))
    #Create a weight vector of size d+1 and set initial values to 0 this will be our best running weight
    w = np.zeros(d+1)

    #Get initial error
    best_error = errorPer(modified_X_train, y_train, w)

    #store the initial best running weight in our current best running weight vector
    w_hat = list(w)

    #Run 5000 iterations of the Pocket learning algorithm
    for t in range(5000):
        #w_hat is the temporary weights during iteration t
        w_hat = PLA(modified_X_train, y_train, w_hat, d, N)

        #Calculate the error of the model using w_hat
        curr_error = errorPer(modified_X_train, y_train, w_hat)
        #If the error is less than the best running error, set the best error to this value
        #and change the best weights found so far to w_hat
        if( curr_error < best_error):
            w = w_hat
            best_error = curr_error
    return w
            
        
#The Perceptron Learning Algorithm which returns an updated weight vector when it encounters a 
#misclassified data point
def PLA(X_train, y_train, w, d, N):
    for i in range(N):
        if(pred(X_train[i], w) != y_train[i]):
            return np.add(w, y_train[i]*X_train[i])
 
def errorPer(X_train,y_train,w):
    avgError = 0
    numberMisclassified = 0
    N = X_train.shape[0]

    #Loop over all samples in X_train and count the number misclassified
    for i in range(N):
        if(pred(X_train[i], w) !=  y_train[i]):
            numberMisclassified += 1
    
    #Get the average error as the number misclassified divided by the numeber of samples
    avgError = numberMisclassified/N
    return avgError 

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

def pred(X_i,w):
    #if the output of X_i dotted with w is positive, return 1
    if(np.dot(X_i, w) > 0):
        return 1
    #otherwise the prediction is -1
    else:
        return -1

def test_SciKit(X_train, X_test, Y_train, Y_test):
    #create a Perceptron object
    clf = Perceptron(max_iter=5000, tol=None, random_state=0)

    #set the inital weights to 0
    d = X_train.shape[1]
    w = np.zeros(d+1)
    clf.coef_ = w

    #train the model on X_train and Y_train
    clf.fit(X_train, Y_train)

    #obtain the confusion matrix of the model's performance on X_test against Y_test
    result = confusion_matrix(Y_test, clf.predict(X_test))

    return result
    

def test_Part1():
    from sklearn.datasets import load_iris
    X_train, y_train = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X_train[50:],y_train[50:],test_size=0.2)

    #Set the labels to +1 and -1
    y_train[y_train == 1] = 1
    y_train[y_train != 1] = -1
    y_test[y_test == 1] = 1
    y_test[y_test != 1] = -1

    #Pocket algorithm using Numpy
    w=fit_perceptron(X_train,y_train)
    cM=confMatrix(X_test,y_test,w)

    #Pocket algorithm using scikit-learn
    sciKit=test_SciKit(X_train, X_test, y_train, y_test)
    
    #Print the result
    print ('--------------Test Result-------------------')
    print("Confusion Matrix is from Part 1a is: ",cM)
    print("Confusion Matrix from Part 1b is:",sciKit)
    

test_Part1()
