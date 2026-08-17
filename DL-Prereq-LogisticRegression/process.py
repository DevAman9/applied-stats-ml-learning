import numpy as np
import pandas as pd


df = pd.read_csv("ecommerce_data.csv")

print(df.head())

print()
df['time_of_day'].hist()

data = df.to_numpy()

# print(data)

np.random.shuffle(data)

print(f"shape of data originally is {data.shape}")


# X contains all the input features except the user action which is the prediction the program will try to make.
# the user actions which are already there are stored into Y.

X = data[:, :-1]
Y = data[:, -1].astype(np.int32)

N , D = X.shape

print(N, D)

# okay now since the original data has the column time of the day as categorical type. with numbers 0-3 signifying the phase of the day.
# we will relpace them with 4 column columns. one phase each and select the one that the row signifies. so the data stays numeric.
X2 = np.zeros((N, D+3))

print(X2.shape) # therefore (500, 8) where D = 5 and add 4 for the time of the day , -1 taken from the time of the day column

X2[:, :(D-1)] = X[:, :(D-1)] # i copied the first four column as it is.

print(X2)

for n in range(N):
    t = int(X[n, D-1])
    X2[n, t+D-1] = 1

# the above for loop just gets t ranging from 0 to 3 and adds it to D-1 = 4. so if t - 0. means start of the day = asign 1 at the first column D-4


# i am assigning X2 back to X since i realised late that X is easier to work with.

X = X2



# splitting data in train and test
Xtrain = X[:-100]
Ytrain = Y[:-100]


Xtest = X[-100:]
Ytest = Y[-100:]

for i in (1, 2):
    m = Xtrain[:, i].mean()
    s = Xtrain[:, i].std()
    Xtrain[: , i] = (Xtrain[: , i] - m) / s
    Xtest[:, i] = (Xtest[: , i] - m) / s

    # return Xtrain, Ytrain, Xtest, Ytest

print(Xtrain.shape, Ytrain.shape, Xtest.shape, Ytest.shape)
# (400, 8) (400,) (100, 8) (100,)


# since i am only doing binary regression right now. i will get the data for output as just 0 and 1.
def get_binary_data():
    X2train = Xtrain[Ytrain <= 1]
    Y2train = Ytrain[Ytrain <= 1]
    X2test = Xtest[Ytest <=1]
    Y2test = Ytest[Ytest <= 1]

    return X2train, Y2train, X2test, Y2test


X2train, Y2train, X2test, Y2test = get_binary_data()

print(X2train.shape, Y2train.shape, X2test.shape, Y2test.shape)


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

