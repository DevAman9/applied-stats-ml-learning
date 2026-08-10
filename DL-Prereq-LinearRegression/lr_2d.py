import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# loading the data

X = []
Y = []

for line in open('data_2d.csv'):
    x1, x2, y = line.split(',')
    X.append([float(x1), float(x2), 1]) # add the bias term
    Y.append(float(y))


X = np.array(X)
Y = np.array(Y)

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

# okay so we have stored x1 in the first coloumn of list X and x2 in the second column of list X. and Y is just the list Y.
# thats why below we get all the rows in X in the first column for x1 and repeat for second column storing x2. and the last one is just Y which we get all of without any splicing

ax.scatter(X[:,0], X[:,1], Y)
plt.show()

# calculating weights.

w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
Yhat = np.dot(X, w)

# compute rsquared

d1 = Y - Yhat
d2 = Y - Y.mean()
r2 = 1 - d1.dot(d1) / d2.dot(d2)

print(r2)
# r squared printed at 0.9980040612475778. which is very close to 1. meaning the model is very very accurate.

