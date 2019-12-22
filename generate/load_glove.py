# Script to load GloVe embeddings

import numpy as np
import bcolz
import pickle

glove_path = '/mnt/c/Users/Ryan/Documents/Linguistics/nlp/embeddings/glove.6B'

words = []
idx = 0
word2idx = {}
vectors = bcolz.carray(np.zeros(1), rootdir =f'{glove_path}/glove.6B.200.dat', mode = 'w')

with open(f'{glove_path}/glove.6B.200d.txt', 'rb') as f:
    for l in f:
        line = l.decode().split()
        word = line[0]
        words.append(word)
        word2idx[word] = idx
        idx += 1
        vect = np.array(line[1:]).astype(np.float)
        vectors.append(vect)

# Save the data

vectors = bcolz.carray(vectors[1:].reshape((400000, 200)), 
            rootdir=f'{glove_path}/6B.200.dat', mode='w')
vectors.flush()
pickle.dump(words, open(f'{glove_path}/6B.200_words.pkl', 'wb'))
pickle.dump(word2idx, open(f'{glove_path}/6B.200_idx.pkl', 'wb'))

# Create the dictionary

vectors = bcolz.open(f'{glove_path}/6B.200.dat')[:]
words = pickle.load(open(f'{glove_path}/6B.200_words.pkl', 'rb'))
word2idx = pickle.load(open(f'{glove_path}/6B.200_idx.pkl', 'rb'))

glove = {w: vectors[word2idx[w]] for w in words}

#print(glove['the'])
