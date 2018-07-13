from numpy import genfromtxt
import numpy as np
from sklearn import datasets, linear_model

datapath = r"C:\Users\sunerhan\Documents\dl-practice\Regression\DummyDeliveryDone.csv"
deliveryData = genfromtxt(datapath, delimiter=',')

print("data",deliveryData)

X = deliveryData[:, :-1]  #所有行，不包括最后列
Y = deliveryData[:, -1]  #所有行，只包括最后列
print("X:",X)
print("Y:",Y)

regr = linear_model.LinearRegression()
regr.fit(X, Y)

print("coefficient",regr.coef_)
print("intercept:",regr.intercept_)
