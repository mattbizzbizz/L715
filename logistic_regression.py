# Reference:
#     https://numpy.org/

from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

# Calculate y^
#     (Z) y
#     return y^
def softmax(Z):

    ret = [] # y^

    # Iterate through values in y
    #     (i) Calculate y^
    for i, z in enumerate(Z):
        ret.append(np.exp(z) / np.sum([np.exp(z_i) for z_i in Z]))

    return np.array(ret)

#def cross_entropy_loss(X, Y, W, B):
#    S = sigmoid(np.dot(X, W) + B)
#    return -(Y * np.log(S) + (1 - Y) * np.log(1 - S))

# Do logistic regression for data
#     (X) Data points
#     (Y) Expected output
#     (W) Weights
#     (B) Bias
#     (A) Learning rate
#     (E) Number of epochs
#     return array of losses across epochs
def logistic_regression(X, Y, W, B, A, E):

    m = len(Y) # Number of data points
    indices = list(range(m + 1)) # Indices of data points
    costs = [] # Array of losses across epochs
    XX = np.concatenate((X, np.ones((X.shape[0], 1))), axis = 1) # X with 1's in last row for partial derivative of b

    # Iterate through epochs to:
    #     (i) Calculate and store cost
    #     (ii) Update weight
    for e in range(E):

        np.random.shuffle(indices) # Randomize indices
        diff = softmax(np.array(np.dot(X, W) + B)) - Y # Calculate softmax(y^) - y
        loss = np.empty([m + 1]) # Array of loss

        # Iterate through randomized indices
        #     (i) Calculate loss
        #     (ii) Update weight
        for i in indices:
            print('diff:', diff)
            print('XX[i]', XX[i])
            loss[i] = [diff * XX_i for XX_i in XX[i]]
            W -= A * loss[i]

        costs.append(loss) # Append loss matrix to costs

    return costs

X, y = datasets.make_blobs(n_samples=100, n_features=3, centers=5, cluster_std=1.05, random_state=3) # Data points and expected output, respectively

weights = np.array(np.random.randn(len(X[0]))) # Weights
biases = np.array(np.random.randn(len(X))) # Biases

logistic_regression(X, y, weights, biases, 0.1, 10)

#fig = plt.figure()
#ax = fig.add_subplot(projection='3d')
#ax.scatter(X[:,0], X[:,1], X[:,2], c=y)
#plt.show()

#for x, y in list(zip(X, y))[:10]:
#    print(x, y)
