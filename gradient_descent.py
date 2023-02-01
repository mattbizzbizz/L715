import numpy as np
from matplotlib import pyplot as plt

np.random.seed(3)

epochs = 50
alphas = [0.1, 0.01, 0.001]

#x = np.random.randn(100, 1)
#y = 4 + 3 * x + np.random.randn(100, 1)
x = np.random.randn(5, 1)
y = 4 + 3 * x + np.random.randn(5, 1)
weights = np.random.randn(1, len(alphas))
m =  len(x)
costs = []
costs_alpha = []

for num_alpha, alpha in enumerate(alphas):
    for epoch in range(epochs):
        y_hat = x * weights[0,num_alpha]
        print(y_hat)
        costs.append(0.5 * m * np.sum(np.square(y_hat - y)))
        print(sum(np.multiply(x, np.subtract(y_hat, y))))
        weights[0, num_alpha] = weights[0, num_alpha] - alpha * np.sum(np.multiply(x, np.subtract(y_hat, y))) / m
    costs_alpha.append(costs)

fi, ax = plt.subplots()
#ax.scatter(x, y)
for num_alpha_2, alpha in enumerate(alphas):
    ax.plot(range(epochs), costs_alpha[num_alpha_2])
plt.show()
