# References:
#   https://numpy.org/doc/stable/reference/generated/

import numpy as np
from matplotlib import pyplot as plt

#def gradient_descent(W, D, A, E, S, M):
#
#    C = []
#    X = np.array([row[0] for row in D])
#    Y = np.array([row[1] for row in D])
#    Y_hat = []
#
#    for e in range(E):
#
#        if S:
#            np.random.shuffle(D)
#            X = np.array([row[0] for row in D])
#            Y = np.array([row[1] for row in D])
#
#
#        if S or M > 1:
#            for i in range(0, len(Y), M):
#                Y_hat = X[i:i+M] * W
#                W = W - A * np.sum((X[i:i+M] * (Y_hat - Y[i:i+M]))) / len(X)
#                C.append(0.5 / len(X) * np.sum(np.square(Y_hat - Y)))
#            C = np.sum(C)
#        else:
#            Y_hat = X * W
#            W = W - A * np.sum(np.multiply(X, Y_hat - Y)) / len(X)
#            C.append(0.5 / len(X) * np.sum(np.square(Y_hat - Y)))
#
#    return C

def gradient_descent(Weight, Data, Alpha, Epochs):

    m = len(Data)
    X = np.array([row[0] for row in Data])
    Y = np.array([row[1] for row in Data])
    cost = []

    for e in range(Epochs):
        diff = X * Weight - Y
        cost.append(np.dot(diff, diff) / (2 * m))
        Weight -= Alpha * np.dot(X, diff) / m

    return cost

def minibatched_gradient_descent(Weight, Data, Alpha, Epochs):

    m = len(Data)
    X = np.array([row[0] for row in Data])
    Y = np.array([row[1] for row in Data])
    batch_size = 5
    cost = []

    for e in range(Epochs):

        diff = X * Weight - Y
        cost.append(np.dot(diff, diff) / (2 * m))

        for batch in range(0, m, batch_size):
            diff = X[batch:batch + batch_size] * Weight - Y[batch:batch + batch_size]
            Weight -= Alpha * np.dot(X[batch:batch + batch_size], diff) / m

    return cost

np.random.seed(3)

epochs = 500
alphas = [0.1, 0.01, 0.001]

x = np.random.randn(100, 1)
y = 4 + 3 * x + np.random.randn(100, 1)
data = np.concatenate((x, y), axis = 1)
weight = np.random.rand()
costs = {}
costs_minibatch = {}

for alpha in alphas:
    costs[alpha] = gradient_descent(weight, data, alpha, epochs)
for alpha in alphas:
    costs_minibatch[alpha] = minibatched_gradient_descent(weight, data, alpha, epochs)


fi, ax = plt.subplots()
for alpha in costs:
    ax.plot(range(epochs), costs[alpha])
    ax.plot(range(epochs), costs_minibatch[alpha])
ax.legend(["\u03B1: 0.1", "\u03B1: 0.01", "\u03B1: 0.001", "\u03B1: 0.1 (Minibatch)", "\u03B1: 0.01 (Minibatch)", "\u03B1: 0.001 (Minibatch)"])
plt.show()
