from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

X, Y = datasets.make_blobs(n_samples=100, n_features=3, centers=5, cluster_std=1.05, random_state=3) # Data points and expected output, respectively

np.random.seed(3)

W = np.random.rand(3, 5)
B = np.random.rand(5)

alpha = 0.001
epochs = 1000

def one_hot_encoding(Y):
    classes = list(set(Y))
    onehot = np.zeros((len(Y), len(classes)))

    for i, v in enumerate(Y):
        onehot[i][v] = 1
    return onehot

def softmax(Z):
    return np.exp(Z) / sum(np.exp(Z))

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def predict(X, W, B):
    return softmax((X @ W) + B)

def cost_function(Y_hat, Y_ohe):
    m = len(Y_ohe)
    Y_log_Y_hat = np.log(y_hat) * Y_ohe
    loss = -np.sum(Y_log_Y_hat) / m
    return loss

def partial_derivative_w(X, diff):
    return np.transpose(np.transpose(diff) @ X)

def partial_derivative_b(diff):
    return np.sum(diff, axis = 0)

def update_hyperparameters(W, B, A, X, Y_ohe):

    m = len(X)

    y_hat = predict(X, W, B)

    cost = cost_function(y_hat, Y_ohe)
    diff = y_hat - Y_ohe

    partial_der_w = partial_derivative_w(X, diff)
    partial_der_b = partial_derivative_b(diff)

    W -= alpha * partial_der_w / m
    B -= alpha * partial_der_b / m

    return W, B, cost, y_hat

def accuracy_metric(Y, y_hat):
    m = len(Y)
    correct = 0

    acc_

    for y_true, y_pred in zip(Y, y_hat):

        y_pred = np.argmax(y_pred)

        if y_pred == y_true:
            correct += 1

    return correct * 100 / m

cost_per_epoch = []
acc_list = []

for epoch in range(epochs):
    print(f'------This is epoch {epoch + 1}------')
    print(f'-- Old weights: {W}')
    print(f'-- Old bias: {B}')

    Y_ohe = one_hot_encoding(Y)
    W, B, cost, y_pred = update_hyperparameters(W, B, alpha, X, Y_ohe)
    acc = accuracy_metric(Y, y_pred)
    acc_list.append(acc)
    cost_per_epoch.append(cost)

    print(f'-- New weights: {W}')
    print(f'-- New bias: {B}')
    print(f'-- Accuracy: {acc}')
    print('\n')

fig = plt.figure()
#ax = fig.add_subplot(projection='3d')
ax = fig.add_subplot()
ax.plot(range(epochs), cost_per_epoch)
#ax.scatter(X[:,0], X[:,1], X[:,2], c=Y)
plt.show()
