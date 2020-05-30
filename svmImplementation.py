import pandas
from sklearn import svm
import csv
import matplotlib.pyplot as plt  # To visualize
from sklearn.linear_model import LinearRegression

def readCVS():
    data = pandas.read_csv("covid_19_data.csv")
    data.head

    print(data)

    return data
if __name__ == '__main__':
    print ("SVM IMPLEMENTATION COCK BIG 19")
    data = readCVS()

    X = data.iloc[:, 5].values.reshape(-1, 1)  # values converts it into a numpy array
    Y = data.iloc[:, 6].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
    print("CONFIRMED")
    print(X)
    print("DEATHS")
    print(Y)


    regr = svm.SVR()
    regr.fit(X,Y)
    regr.predict([[1,1]])


    linear_regressor = LinearRegression()  # create object for the class
    linear_regressor.fit(X, Y)  # perform linear regression
    Y_pred = linear_regressor.predict(X)  # make predictions
    plt.scatter(X, Y)
    plt.plot(X, Y_pred, color='red')
    plt.show()