# Reference:
#     https://numpy.org/

from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def softmax(Z):

    ret = []

    for i, z in enumerate(Z):
        ret.append(np.exp(z) / np.sum([np.exp(z_i) for z_i in Z]))

    return np.array(ret)

def cross_entropy_loss(X, Y, W, B):
    S = sigmoid(np.dot(X, W) + B)
    return -(Y * np.log(S) + (1 - Y) * np.log(1 - S))


# Calculate costs for graident descent
def stocastic_gradient_descent(X, Y, W, A, E):

    m = len(X) # Number of training examples
    cost = [] # Costs
    indices = list(range(len(X)))

    # Iterate through epochs to:
    #     (i) Calculate and store cost
    #     (ii) Update weight
    for e in range(E):

        np.random.shuffle(indices) # Randomize indices

        for i in indices:

            diff = X[i] * W - Y[i]
            cost.append(np.dot(diff, diff) / (2 * m))
            W -= A * np.dot(X[i], diff) / m

    return cost

def logistic_regression(X, Y, W, B, A, E):
    return softmax(np.array(np.dot(X, W) + B))

X, y = datasets.make_blobs(n_samples=100, n_features=3, centers=5, cluster_std=1.05, random_state=3)

weights = np.array(np.random.randn(len(X[0])))
biases = np.array(np.random.randn(len(X)))

#fig = plt.figure()
#ax = fig.add_subplot(projection='3d')
#ax.scatter(X[:,0], X[:,1], X[:,2], c=y)
#plt.show()

#for x, y in list(zip(X, y))[:10]:
#    print(x, y)
