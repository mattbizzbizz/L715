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

from matplotlib import pyplot as plt, lines, animation

weights = np.random.rand(2) # list of weights
df = [] # list perceptron training data
df_pos = [] # list of perceptron training data with 1 output values
df_neg = [] # list of perceptron training data with 0 output values
epochs = 25 # number of epochs to run
bias = np.random.rand() # value for the bias
alpha = 0.001 # learning rate
decision_boundaries = []

# Open .tsv file
with open('perceptron_training_data.tsv') as file:
    next(file)
    for line in file:
        l = line.strip('\n').split('\t')
        l[0] = float(l[0])
        l[1] = float(l[1])
        l[2] = int(l[2])
        df.append(l)

for line in df:
    if line[2]:
        df_pos.append(line)
    else:
        df_neg.append(line)

minx = min([row[0] for row in df]) # the minimum value for feature 1
maxx = max([row[0] for row in df]) # the maximum value for feature 1
miny = min([row[1] for row in df]) # the minimum value for feature 2
maxy = max([row[1] for row in df]) # the maximum value for feature 2

def get_decision_boundary():
    global decision_boundaries
    y0 = (-bias - minx * weights[0]) / weights[1]
    y1 = (-bias - maxx * weights[1]) / weights[1]
    decision_boundaries.append([y0, y1])

get_decision_boundary()

for epoch in range(epochs):

    correct = 0

    for row in df:

        x = [row[0], row[1]]
        y_hat = 0.0

        if (np.dot(x, weights) + bias) > 0:
            y_hat = 1.0

        if y_hat == row[2]:
            correct += 1
        else:
            if y_hat == 0:
                weights = np.add(weights, [alpha * value for value in x])
                bias += alpha
            elif y_hat == 1:
                weights = np.subtract(weights, [alpha * value for value in x])
                bias -= alpha

    get_decision_boundary()

    accuracy = correct/len(df) * 100
    print('=========EPOCH========= {}'.format(epoch + 1))
    print('Accuracy: {}%'.format(accuracy))
    print('Weights:', weights)
    print('Bias:', bias)
    print('Decision Boundary:', decision_boundaries[-1])
    print()


    if correct == len(df):
        break

fig, ax = plt.subplots()
def acb(i):
    ax.clear()
    ax.set_xlim([minx - 0.5, maxx + 0.5])
    ax.set_ylim([miny - 0.5, maxy + 0.5])
    ax.set_title(f'Epoch{i+1}')
    ax.scatter([j[0] for j in df_pos], [k[1] for k in df_pos], c = 'red')
    ax.scatter([j[0] for j in df_neg], [k[1] for k in df_neg], c = 'blue')
    ax.plot([minx, maxx], decision_boundaries[i])
anim = animation.FuncAnimation(fig, acb, frames = len(decision_boundaries))
plt.show()
