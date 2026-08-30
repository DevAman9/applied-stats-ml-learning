import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

# i am using the breast cancer data from sklearn datasets.

data = load_breast_cancer()

X = data.data
y = data.target

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Classes:", np.unique(y))


# using a 80/20 split.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# getting the data and setting it up.
# i am making empty matrices for means, variances and priors (not a matrix). so i can later store the mean, var and p values in there.
class MyGaussianNB:

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.classes = np.unique(y)
        n_classes = len(self.classes)

        self.means = np.zeros((n_classes, n_features))
        print(f"means array{self.means.shape}")
        self.variances = np.zeros((n_classes, n_features))
        self.priors = np.zeros(n_classes)

        # calculating the means, variance and p values for each of the classes separately.

        for idx, c in enumerate(self.classes):
            X_c = X[y == c]

            self.means[idx] = X_c.mean(axis=0)
            self.variances[idx] = X_c.var(axis=0)
            self.priors[idx] = X_c.shape[0] / n_samples


    def predict(self, X):
        predictions = []
        for i, x in enumerate(X):
            predictions.append(self._predict_one(x))

        return predictions



    # easy stuff.

    def _predict_one(self, x):
        scores = []

        for idx, c in enumerate(self.classes):
            prior = self.priors[idx]
            means = self.means[idx, :]
            vars = self.variances[idx, :]

            densities = self._gaussian_pdf(x, means, vars)

            score = np.log(prior) + np.sum(np.log(densities))

            scores.append(score)

        return self.classes[np.argmax(scores)]

    # this is the gaussian formula i used to calculated the density or basically to score the x value in the feature's gaussian/normal distribution.
    def _gaussian_pdf(self, x, mean, var):
        numerator = np.exp(-((x-mean)**2)/(2*(var)))
        denominator = np.sqrt(2 * np.pi * (var))
        return numerator/denominator


model = MyGaussianNB()

model.fit(X_train, y_train)

print(model.classes)
print(model.means.shape)
print(model.variances.shape)
print(model.priors)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
y_pred = model.predict(X_test)

print("\nMy Gaussian NB accuracy:")
print(accuracy_score(y_test, y_pred))


sk_model = GaussianNB()

sk_model.fit(X_train, y_train)

sk_pred = sk_model.predict(X_test)

# also testimg the accuracy compared to the sklearn guassian naive bayes model accuracy.

print("\nSklearn GaussianNB accuracy:")
sklearnaccuracy = accuracy_score(y_test, sk_pred)
print(sklearnaccuracy)
change_ = (accuracy_score(y_test, y_pred) / sklearnaccuracy) * 100
print(f"my accuracy compared to sklearn was {change_}%")