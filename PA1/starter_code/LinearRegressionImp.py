import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn import linear_model
from sklearn.metrics import mean_squared_error


def fit_LinRegr(X_train, y_train):
   #The number of samples we have is stored in N
    N = X_train.shape[0]
    #The dimensionality of each data point is stored in d
    d = X_train.shape[1]
    #Create a column of ones and append it to X_train
    ones = np.ones((N,1))
    modified_X_train = np.hstack((ones, X_train))

    #Compute the pseudo-inverse of modified_X_train:
    X_transpose_X = np.matmul(np.transpose(modified_X_train), modified_X_train)
    pseudo_inverse_X = np.matmul(np.linalg.pinv(X_transpose_X), np.transpose(modified_X_train))

    #Use pseudo-inverse and y-train to calculate weights w
    w = np.matmul(pseudo_inverse_X, y_train)

    return w

def mse(X_train,y_train,w):
    #The number of samples we have is stored in N
    N = X_train.shape[0]
    #The dimensionality of each data point is stored in d
    d = X_train.shape[1]
    #Create a column of ones and append it to X_train
    ones = np.ones((N,1))
    modified_X_train = np.hstack((ones, X_train))

    #Calculate sum of squared error
    error_sum = 0
    for i in range(N):
       error_sum += pow(pred(modified_X_train[i], w) - y_train[i], 2)

    #Divide by number of samples to get average error
    avgError = error_sum/N

    return avgError
    

def pred(X_i,w):
    return np.dot(X_i, w)

def test_SciKit(X_train, X_test, Y_train, Y_test):
    #create linear regression model and train it on X_train and Y_train
    clf = linear_model.LinearRegression()
    clf.fit(X_train, Y_train)

    #return mean squared error of classifier using X_test against Y_test
    return mean_squared_error(Y_test, clf.predict(X_test))


def subtestFn():
    # This function tests if your solution is robust against singular matrix

    # X_train has two perfectly correlated features
    X_train = np.asarray([[1, 2], [2, 4], [3, 6], [4, 8]])
    y_train = np.asarray([1,2,3,4])
    
    try:
      w=fit_LinRegr(X_train, y_train)
      print ("weights: ", w)
      print ("NO ERROR")
    except:
      print ("ERROR")

def testFn_Part2():
    X_train, y_train = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X_train,y_train,test_size=0.2)
    
    w=fit_LinRegr(X_train, y_train)
    
    #Testing Part 2a
    e=mse(X_test,y_test,w)
    
    #Testing Part 2b
    scikit=test_SciKit(X_train, X_test, y_train, y_test)
    
    print("Mean squared error from Part 2a is ", e)
    print("Mean squared error from Part 2b is ", scikit)

print ('------------------subtestFn----------------------')
subtestFn()

print ('------------------testFn_Part2-------------------')
testFn_Part2()


#My implementation achieves an error that is within 0.000000000001 of the existing modules in the scikit-learn library