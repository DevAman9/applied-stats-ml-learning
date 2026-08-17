import numpy as np
import matplotlib.pyplot as plt

N = 100
D = 2

X = np.random.randn(N, D)

X[:50, :] = X[:50, :] - 2*np.ones((50, D))
X[50: , :] = X[50:,:] + 2*np.ones((50, D))

T = np.array([0]*50 + [1]*50)

# making an array of ones, to act as dummy weight for the bias. transposing is important so the internal dimensions match while dot product
ones = np.ones((N, 1))
print(ones.shape, X.shape)

Xb = np.concatenate((ones, X), axis = 1)

w = np.random.randn(D+1)

z = Xb.dot(w)

def sigmoid(z):
    return 1/(1+np.exp(-z))

Y = sigmoid(z)


def cross_entropy(Y, T):
    E = 0
    for i in range(N):
        if T[i] == 1:
            E -= np.log(Y[i])
        else:
            E -= np.log(1 - Y[i])

    return E

print(cross_entropy(Y, T))

w1 = np.array([0, 4, 4])

plt.scatter(X[:, 0], X[:, 1], c = T, s = 100, alpha = 0.5)

x_axis = np.linspace(-6, 6, 100)
y_axis = -x_axis

plt.plot(x_axis, y_axis)
plt.show()


z1 = Xb.dot(w1)
Y1 = sigmoid(z1)

print(cross_entropy(Y1, T))