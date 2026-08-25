
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
iris = datasets.load_iris()

X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=1234
)

# print(X_train.shape)  # (120, 4) so we have four dimensions and
# print(X_train[0])
# print(y_train.shape)
# print(y_train)

plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', s=20)
plt.show()


a = [1, 1, 1, 1, 2, 2, 3, 4, 5, 6]

from collections import Counter

most_common = Counter(a).most_common()
# print(most_common[0][0])


from knn import KNN

clf = KNN(k=10)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

acc = np.sum(predictions == y_test) / len(y_test)
print(acc)