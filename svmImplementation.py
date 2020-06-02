import pandas
from sklearn import svm

import sklearn
import csv
import matplotlib.pyplot as plt  # To visualize
from sklearn.linear_model import LinearRegression

import numpy as np




def readCVS():
    data = pandas.read_csv("covid_19_data.csv")
    data.head

    print(data)

    return data

def noSeQuePedo():
    data = readCVS()
    ID = data.iloc[:, 5].values.reshape(-1, 1)  # values converts it into a numpy array
    X = data.iloc[:, 5].values.reshape(-1, 1)  # values converts it into a numpy array
    Y = data.iloc[:, 6].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
    print("ID")
    print(ID)
    print("CONFIRMED")
    print(X)
    print("DEATHS")
    print(Y)

    regr = svm.SVR()
    regr.fit(X, Y)
    regr.predict([[1, 1]])

    linear_regressor = LinearRegression()  # create object for the class
    linear_regressor.fit(X, Y)  # perform linear regression
    Y_pred = linear_regressor.predict(X)  # make predictions
    plt.scatter(X, Y)
    plt.plot(X, Y_pred, color='red')
    plt.show()

def svm():
    print("SVM IMPLEMENTATION COCK BIG 19")
    print("https://scikit-learn.org/stable/modules/svm.html#svm-classification")
    print("https://www.kaggle.com/sudalairajkumar/novel-corona-virus-2019-dataset/data?select=covid_19_data.csv")
    data = readCVS()
    ID = data.iloc[:, 0].values.reshape(-1, 1)  # values converts it into a numpy array
    X = data.iloc[:, 5].values.reshape(-1, 1)  # values converts it into a numpy array
    Y = data.iloc[:, 6].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
    print("ID")
    print(ID)
    print("CONFIRMED")
    print(X)
    print("DEATHS")
    print(Y)
    infectados = 0
    for x in X:
        infectados += x[0]
    muertos = 0
    for x in Y:
        muertos += x[0]

    print(infectados)
    print(muertos)
    newY = []
    for x in Y:
        #print(x[0])
        newY.append(int(x))
    #print(newY)

    newX = []
    for x in X:
        newX.append(int(x))
    #print(newX)

    regr = sklearn.svm.SVR()
    regr.fit(ID, X)

    ypred = regr.predict([[34215]])
    print(ypred)

    ypred = regr.predict(X)
    print(ypred)


    plt.plot(ID, X, color='red')
    plt.show()
    print("END")
if __name__ == '__main__':
    svm()