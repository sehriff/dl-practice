from numpy import genfromtxt
import numpy as np
from sklearn import datasets, linear_model

datapath = r"C:\Users\sunerhan\Documents\dl-practice\Regression\Delivery.csv"
deliveryData = genfromtxt(datapath, delimiter=',')

print("data",deliveryData)

X = deliveryData[:, :-1]  #所有行，不包括最后列
Y = deliveryData[:, -1]  #所有行，只包括最后列
print("X:",X)
print("Y:",Y)

regr = linear_model.LinearRegression()
regr.fit(X, Y)

print("coefficient",regr.coef_,"intercept:",regr.intercept_)

# xPred = np.array([102, 6]).reshape(1, -1)
xPred = np.reshape([102,6],(1, -1))
yPred = regr.predict(xPred)
print("predcted y:",yPred)