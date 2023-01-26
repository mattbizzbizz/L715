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

import matplotlib as mpl
import matplotlib.pyplot as plt

weights = [] # list of weights
df = [] # list perceptron training data
epochs = 25 # number of epochs to run
bias = np.random.rand() # value for the bias
alpha = 0.001 # learning rate

# Open .tsv file
with open('perceptron_training_data.tsv') as file:
    next(file)
    for line in file:
        l = line.strip('\n').split('\t')
        l[0] = float(l[0])
        l[1] = float(l[1])
        l[2] = int(l[2])
        df.append(l)


df_pos = []
df_neg = []
for line in df:
    if line[2]:
        df_pos.append(line)
    else:
        df_neg.append(line)

minx = min([row[0] for row in df]) 
maxx = max([row[0] for row in df]) 
miny = min([row[1] for row in df])
maxy = max([row[1] for row in df])

weights.append(np.random.rand())
weights.append(np.random.rand())

for i in range(epochs):

    print("Epoch:", i + 1)
    count = 0

    for row in df:

        x = [row[0], row[1]]
        y_hat = 0.0

        if (np.dot(x, weights)) > 0:
            y_hat = 1.0

        if y_hat == row[2]:
            count += 1
        else:
            if y_hat == 0:
                weights = np.add(weights, [alpha * value for value in x])
                bias += alpha
            elif y_hat == 1:
                weights = np.subtract(weights, [alpha * value for value in x])
                bias -= alpha

    print(weights)

    fig, ax = plt.subplots()
    ax.set_xlim([minx - 0.5, maxx + 0.5])
    ax.set_ylim([miny - 0.5, maxy + 0.5])
    ax.set_title(f'Epoch {i}')
    ax.scatter([j[0] for j in df_pos], [k[1] for k in df_pos], c = 'red')
    ax.scatter([j[0] for j in df_neg], [k[1] for k in df_neg], c = 'blue')
    y0 = (-bias - minx * weights[0]) / weights[1]
    y1 = (-bias - maxx * weights[1]) / weights[1]
    ax.plot([minx, maxx], [y0, y1])
    plt.show()

    print("Number of correctly classified examples:", count)
    if count == len(df):
        break
