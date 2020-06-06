import pandas
from openpyxl.utils import dataframe
from sklearn import svm

import sklearn
import csv
import matplotlib.pyplot as plt  # To visualize
from sklearn.linear_model import LinearRegression

import numpy as np




def readCVS():
    data = pandas.read_csv("covid_19_data.csv")
    data.head
    data = data[data["Country/Region"].str.contains("US")]


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
    CONFIRMED = data.iloc[:, 5].values.reshape(-1, 1)  # values converts it into a numpy array
    DEATHS = data.iloc[:, 6].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
    print("ID")
    print(ID)
    print("CONFIRMED")
    print(CONFIRMED)
    print("DEATHS")
    print(DEATHS)
    infectados = 0
    for x in CONFIRMED:
        infectados += x[0]
    muertos = 0
    for x in DEATHS:
        muertos += x[0]

    print(infectados)
    print(muertos)
    newY = []
    for x in DEATHS:
        #print(x[0])
        newY.append(int(x))
    #print(newY)

    newX = []
    for x in CONFIRMED:
        newX.append(int(x))
    #print(newX)
    print("SIZE OF")
    print(len(ID))

    newID = []
    for i in range(0, len(ID)):
        newID.append([i])

    regrCases = sklearn.svm.SVR()
    regrCases.fit(newID, CONFIRMED)

    ypred = regrCases.predict([[5750]])
    print("CASES: PREDICTION FOR ID 5750",ypred)
    ypred = regrCases.predict([[5751]])
    print("CASES: PREDICTION FOR ID 5751",ypred)
    ypred = regrCases.predict([[5780]])
    print("CASES PREDICTION FOR ID 5780",ypred)

    prediccionCASOS50dias =[]

    for i in range(0,50):
        prediccionCASOS50dias.append(regrCases.predict([[5744 + 1]]))

    print(prediccionCASOS50dias)


    regrDeaths = sklearn.svm.SVR()
    regrDeaths.fit(newID, DEATHS)

    ypred = regrDeaths.predict([[5750]])
    print("DEATHS: PREDICTION FOR ID 5750",ypred)
    ypred = regrDeaths.predict([[5751]])
    print("DEATHS: PREDICTION FOR ID 5751",ypred)
    ypred = regrDeaths.predict([[5780]])
    print("DEATHS PREDICTION FOR ID 5780",ypred)

    prediccionMuertes50Dias =[]

    for i in range(0,50):
        prediccionMuertes50Dias.append(regrDeaths.predict([[5744 + i]]))
    print(prediccionMuertes50Dias)

    plt.plot(newID, CONFIRMED, color='blue', label = "CASES")
    plt.plot(newID, DEATHS, color='red', label = "DEATHS")
    plt.legend(loc = 'upper left', frameon = False)
    plt.show()
    sig50 =[]
    for i in range(50):
        sig50.append(i)

    plt.plot(sig50, prediccionCASOS50dias, color='blue', label="CASES")
    plt.legend(loc='upper left', frameon=False)
    plt.show()
    plt.plot(sig50, prediccionMuertes50Dias, color='red', label="DEATHS")
    plt.legend(loc='upper left', frameon=False)
    plt.show()

    print("END")
if __name__ == '__main__':
    svm()