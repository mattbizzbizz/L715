# References:
#   https://numpy.org/doc/stable/reference/generated/

import numpy as np
from matplotlib import pyplot as plt

def gradient_descent(W, D, A, E, S, M):

    C = []
    X = np.array([row[0] for row in D])
    Y = np.array([row[1] for row in D])

    for e in range(E):

        if S:
            np.random.shuffle(D)
            X = np.array([row[0] for row in D])
            Y = np.array([row[1] for row in D])

        Y_hat = X * W

        C.append(0.5 / len(X) * np.sum(np.square(Y_hat - Y)))

        if S or M > 0:
            for i in range(0, len(Y), M):
                W = W - A * np.sum((X[i:i+M] * (Y_hat[i:i+M] - Y[i:i+M]))) / len(X)
        else:
            W = W - A * np.sum(np.multiply(X, Y_hat - Y)) / len(X)

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
    costs.append(gradient_descent(weight, data, alpha, epochs, False, 1))
costs.append(gradient_descent(weight, np.copy(data), 0.1, epochs, True, 1))
costs.append(gradient_descent(weight, np.copy(data), 0.1, epochs, False, 5))

linestyles = ['solid', 'solid', 'solid', 'dashed', 'dotted']

fi, ax = plt.subplots()
for num, linestyle in zip(range(len(costs)), linestyles):
    ax.plot(range(epochs), costs[num], linestyle = linestyle)
ax.legend(["\u03B1: 0.1", "\u03B1: 0.01", "\u03B1: 0.001", "\u03B1: 0.001 (stocastic)", "\u03B1: 0.001 (minibatch = 5)"])
plt.show()
