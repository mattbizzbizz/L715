import re 
import numpy as np

test_text = '''where are you?
is she in mexico?
i am in greece.
she is in mexico.
is she in england?
'''

train_text = '''are you still here?
where are you?
he is in mexico.
are you tired?
i am tired.
are you in england?
were you in mexico?
is he in greece?
were you in england?
are you in mexico?
i am in mexico.
are you still in mexico? 
are you in greece again?
she is in england.
he is tired.
'''

def tokenise(s):
    return [i for i in re.sub('([.?])', ' \g<1>', s).strip().split(' ') if i]

def one_hot(y, classes):
    onehot = np.zeros((len(y), classes)) # creates matrix of ? rows, ? columns 
    
    # Iterate through y and update onehot's column to 1 based on the class
    # y [0, 1, 4, 3, 2]
    for i, v in enumerate(y):
        onehot[i][v] = 1
    return onehot

def relu(X):
    return np.maximum(0, X)

def softmax(Z):
    return np.exp(Z) / sum(np.exp(Z))

vocab = list(set([token for token in re.sub('([.?])', ' \g<1>', train_text)
             .replace(' ', '\n').strip().split('\n') if token]))
vocab += ['<BOS>', '<EOS>', '<PAD>']
vocab.sort()

word2idx = {word: idx for idx, word in enumerate(vocab)}
idx2word = {idx: word for word, idx in word2idx.items()}

print(word2idx)

pad = max([len(tokenise(i)) for i in train_text.split('\n')]) + 1
train_sentences = []
for line in train_text.strip().split('\n'):
        tokens = tokenise(line)
        padded = ['<BOS>'] + tokens + ['<EOS>'] + ['<PAD>'] * (pad - len(tokens))
        train_sentences.append([word2idx[token] for token in padded])

X = []
y = []

for sentence in train_sentences:
    for i in range(pad - 2):
            X.append([sentence[i], sentence[i+1]])
            y.append(sentence[i+2])
                  
X = np.array(X)
X = np.append(X, np.ones((X.shape[0], 1)), axis = 1)
Y = np.array(y)
Yo = one_hot(Y, len(vocab))

print(X.shape)
print(Y.shape)
print(Yo.shape)

#E = np.random.randn(2, 4)
h = np.random.randn(3, 6) 
o = np.random.randn(6, len(vocab)) 

#print(E.shape)
print(h.shape)
print(o.shape)

hidden_layer = relu(X @ h)
output_layer = softmax(hidden_layer @ o)
cross_entropy_loss = -np.sum(Yo * np.log(output_layer))
print(cross_entropy_loss)
