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

def logistic_regression(X, Y, W, B, A, E):
    Y_hat = softmax(np.array(np.dot(W, X) + B))
    return Y_hat

X, y = datasets.make_blobs(n_samples=100, n_features=3, centers=5, cluster_std=1.05, random_state=3)

weights = np.array(np.random.randn(len(X[0])))
biases = np.array(np.random.randn(len(X)))

#fig = plt.figure()
#ax = fig.add_subplot(projection='3d')
#ax.scatter(X[:,0], X[:,1], X[:,2], c=y)
#plt.show()

#for x, y in list(zip(X, y))[:10]:
#    print(x, y)

#print(X[:10])
#print(softmax(np.dot(X, weights) + biases))

print(softmax([0.6, 1.1, -1.5, 1.2, 3.2, -1.1]))
