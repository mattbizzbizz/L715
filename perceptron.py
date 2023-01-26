# Links used:
#  https://realpython.com/python-command-line-arguments/
#  https://stackoverflow.com/questions/4319236/remove-the-newline-character-in-a-list-read-from-a-file
#  https://numpy.org/doc/1.16/reference/generated/numpy.random.rand.html#numpy.random.rand
#  https://www.geeksforgeeks.org/simple-ways-to-read-tsv-files-in-python/
#  https://www.tutorialspoint.com/generating-random-number-list-in-python
#  https://stackoverflow.com/questions/4796764/read-file-from-line-2-or-skip-header-row
#  https://numpy.org/doc/stable/reference/generated/numpy.dot.html
#  https://stackoverflow.com/questions/53485221/numpy-multiply-array-with-scalar
#  https://stackoverflow.com/questions/5825208/python-array-multiply

import numpy as np
import sys

weights = [] # list of weights
df = [] # list perceptron training data
epochs = int(sys.argv[1]) # number of epochs to run
bias = np.random.rand() # value for the bias
alpha = 0.001 # learning rate

# Open .tsv file
with open('perceptron_training_data.tsv') as file:
    next(file)
    for line in file:
        l = line.strip('\n').split('\t')
        df.append(l)

weights.append(np.random.rand())
weights.append(np.random.rand())


for i in range(epochs):

    print("Epoch:", i + 1)

    count = 0

    for line in df:

        x = [float(line[0]), float(line[1])]
        y_hat = 0.0

        if (np.dot(x, weights)) > 0:
            y_hat = 1.0

        if y_hat == float(line[2]):
            count += 1
        else:

            if y_hat == 0:
                weights = np.add(weights, [alpha * value for value in x])
                bias += alpha
            elif y_hat == 1:
                weights = np.subtract(weights, [alpha * value for value in x])
                bias -= alpha

    print("Number of correctly classified examples:", count)
    if count == len(df):
        print(weights)
        break

    print(weights)
