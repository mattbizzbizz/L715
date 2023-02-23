# References:
#   https://numpy.org/

import numpy as np
from matplotlib import pyplot as plt

# Calculate costs for graident descent
def gradient_descent(Weight, Data, Alpha, Epochs):

    m = len(Data) # Number of training examples
    X = np.array([row[0] for row in Data]) # Inputs
    Y = np.array([row[1] for row in Data]) # Expected Outputs
    cost = [] # Costs

    # Iterate through epochs to:
    #     (i) Calculate and store cost
    #     (ii) Update weight
    for e in range(Epochs):
        diff = X * Weight - Y
        cost.append(np.dot(diff, diff) / (2 * m))
        Weight -= Alpha * np.dot(X, diff) / m

    return cost

# Calculate costs for minibatched graident descent
def minibatched_gradient_descent(Weight, Data, Alpha, Epochs):

    m = len(Data) # Number of training examples
    X = np.array([row[0] for row in Data]) # Inputs
    Y = np.array([row[1] for row in Data]) # Expected Outputs
    batch_size = 5 # Batch size
    cost = [] # Costs

    # Iterate through epochs to:
    #     (i) Calculate and store cost
    #     (ii) Update weight
    for e in range(Epochs):

        diff = X * Weight - Y
        cost.append(np.dot(diff, diff) / (2 * m))

        # Iterate through data by batch size to:
        #     (i) Update weight
        for batch in range(0, m, batch_size):
            diff = X[batch:batch + batch_size] * Weight - Y[batch:batch + batch_size]
            Weight -= Alpha * np.dot(X[batch:batch + batch_size], diff) / batch_size

    return cost


# Calculate costs for stocastic graident descent
def stocastic_gradient_descent(Weight, Data, Alpha, Epochs):

    m = len(Data) # Number of training examples
    batch_size = 5 # Batch size
    cost = [] # Costs

    # Iterate through epochs to:
    #     (i) Calculate and store cost
    #     (ii) Update weight
    for e in range(Epochs):

        np.random.shuffle(Data) # Randomize inputs
        X = np.array([row[0] for row in Data]) # Inputs
        Y = np.array([row[1] for row in Data]) # Expected Outputs

        diff = X * Weight - Y
        cost.append(np.dot(diff, diff) / (2 * m))

        # Iterate through data by batch size to:
        #     (i) Update weight
        for batch in range(0, m, batch_size):
            diff = X[batch:batch + batch_size] * Weight - Y[batch:batch + batch_size]
            Weight -= Alpha * np.dot(X[batch:batch + batch_size], diff) / batch_size

    return cost


np.random.seed(3) # Random seed

alphas = [0.1, 0.01, 0.001] # Learning rates

x = np.random.randn(100, 1) # Inputs
y = 4 + 3 * x + np.random.randn(100, 1) # Expected Outputs
data = np.concatenate((x, y), axis = 1) # X & Y

weight = np.random.rand() # Weight

costs = {} # Costs for gradient descent
costs_minibatch = {} # Cost for minibatched gradient descent
costs_stocastic = {} # Cost for minibatched gradient descent

# Add gradient descent to costs for each learning rate
for alpha in alphas:
    costs[alpha] = gradient_descent(weight, data, alpha, 3000)

# Add minibatched gradient descent to costs for each learning rate
for alpha in alphas:
    costs_minibatch[alpha] = minibatched_gradient_descent(weight, data, alpha, 500)

# Add stocastic gradient descent to costs for each learning rate
for alpha in alphas:
    costs_stocastic[alpha] = stocastic_gradient_descent(weight, data, alpha, 200)

# Plot costs
fi, ax = plt.subplots()
for alpha in alphas:
    #ax.plot(range(3000), costs[alpha])
    #ax.plot(range(500), costs_minibatch[alpha])
    ax.plot(range(200), costs_stocastic[alpha])
plt.xlabel("Epochs")
plt.ylabel("Cost")
plt.legend(["alpha: " + str(alpha) for alpha in alphas])
plt.show()
