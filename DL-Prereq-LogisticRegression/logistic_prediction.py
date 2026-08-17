import numpy as np
# import pandas as pd
from process import get_binary_data

# getting the train data since we dont need the test right now.
X, Y, _, _ = get_binary_data()

# randomly initialising weights.

D = X.shape[1]

W = np.random.randn(D)
b = 0


def sigmoid(a):
    return 1/(1+ np.exp(-a))

def forward(X, W, b):
    return sigmoid(X.dot(W) + b)


P_Y_given_X = forward(X, W, b)
print(P_Y_given_X.shape)
print(P_Y_given_X)

predictions = np.round(P_Y_given_X)
print(predictions)

def classification_rate(Y, P):
    return np.mean(Y == P)


print(classification_rate(Y, predictions))


# ones = np.array([[1]*500]).T
# print(np.shape(ones))
# Xb = np.concatenate((ones, X), axis = 1)
#
# w = np.random.randn(5+1)
#
# z = Xb.dot(w)
#
# def sigmoid(z):
#     return 1/(1+ np.exp(-z))
#
#
#
# print(sigmoid(z))
