# References:
#   https://numpy.org/doc/stable/reference/generated/

import numpy as np
from matplotlib import pyplot as plt

def gradient_descent(W, D, A, E):

    C = []
    X = np.array([row[0] for row in D])
    Y = np.array([row[1] for row in D])

    for e in range(E):
        Y_hat = X * W
        C.append(0.5 / len(X) * np.sum(np.square(Y_hat - Y)))
        W = W - A * np.sum(np.multiply(X, Y_hat - Y)) / len(X)

    return C


def stocastic_gradient_descent(W, D, A, E):

    C = []

    for e in range(E):

        np.random.shuffle(D)
        X = np.array([row[0] for row in D])
        Y = np.array([row[1] for row in D])

        Y_hat = X * W
        C.append(0.5 / len(X) * np.sum(np.square(Y_hat - Y)))

        for i in range(len(Y)):
            W = W - A * (X[i] * (Y_hat[i] - Y[i])) / len(X)

    return C

np.random.seed(3)

epochs = 3000
alphas = [0.1, 0.01, 0.001]

x = np.random.randn(100, 1)
y = 4 + 3 * x + np.random.randn(100, 1)
data = np.concatenate((x, y), axis = 1)
weight = np.random.rand()
costs = []

for alpha in alphas:
    costs.append(gradient_descent(weight, data, alpha, epochs))
for alpha in alphas:
    costs.append(stocastic_gradient_descent(weight, np.copy(data), alpha, epochs))

fi, ax = plt.subplots()
for num in range(len(costs)):
    ax.plot(range(epochs), costs[num])
ax.legend(["\u03B1: " + str(alpha) + s for s in ["", " (Stocastic)"] for alpha in alphas])
plt.show()
