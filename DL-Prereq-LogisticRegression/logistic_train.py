import numpy as np
from process import get_binary_data
import matplotlib.pyplot as plt

Xtrain, Ytrain, Xtest, Ytest = get_binary_data()

# i got the data and now just to start i will just randomly initialise weights.

D = Xtrain.shape[1]

print(f"D is {D}")

W = np.random.randn(D)
b = 0

print(len(Xtrain))

def sigmoid(z):
    return 1/(1+ np.exp(-z))

def forward(Xtrain, W, b):
    return sigmoid(Xtrain.dot(W) + b)

def classification_rate(Y, P):
    return np.mean(Y == P)

def cross_entropy(T, pY):
    return -np.mean(T*np.log(pY) + (1 - T)*np.log(1 - pY))

train_costs = []
test_costs = []
learning_rate = 0.001

for i in range(10000):
    pYtrain = forward(Xtrain, W, b)
    pYtest = forward(Xtest, W, b)

    ctrain = cross_entropy(Ytrain, pYtrain)
    ctest = cross_entropy(Ytest, pYtest)

    train_costs.append(ctrain)
    test_costs.append(ctest)

    # using the formula X.transposed dot ( probabilities - actual )
    W -= learning_rate*Xtrain.T.dot(pYtrain - Ytrain)

    b -= learning_rate*(pYtrain - Ytrain).sum()

    if i % 1000 == 0:
        print(i, ctrain, ctest)


print("Final train classification_rate:", classification_rate(Ytrain, np.round(pYtrain)))
print("Final test classification_rate:", classification_rate(Ytest, np.round(pYtest)))

plt.plot(train_costs, label='train cost')
plt.plot(test_costs, label='test cost')
plt.legend()
plt.show()