import numpy as np
import matplotlib.pyplot as plt

# first i will just be loading up the data into the program
X = []
Y = []

for line in open('data_1d.csv'):
    x, y = line.split(',')
    X.append(float(x))
    Y.append(float(y))


# converting them into numpy

X = np.array(X)
Y = np.array(Y)

# plotting the data

plt.scatter(X, Y)
plt.show()

# applying some equations to calculate a and b

denominator = X.dot(X) - X.mean() * X.sum()
a = ( X.dot(Y) - Y.mean() * X.sum()) / denominator
b = (Y.mean() * X.dot(X) - X.mean() * X.dot(Y)) / denominator

# calculating predicted y

Yhat = a*X + b

plt.scatter(X, Y)
plt.plot(X, Yhat)
plt.show()

# calculating R squared

d1 = Y - Yhat
d2 = Y - Y.mean()
r2 = 1 - d1.dot(d1) / d2.dot(d2)

print(r2)

# the r2 printed as 0.9911838202977805 which confirms that the model is pretty accurate (closer to 1)




