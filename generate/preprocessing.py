# Script for preparing data for generation model

import torch
import torch.nn as nn
from torch import optim

import torchtext
from torchtext import data, datasets

import numpy as np
import unicodedata
import re
import random
import pickle

# Hyperparams
NUM_EPOCHS = 1
HIDDEN_SIZE = 256
LEARNING_RATE = 0.005
BATCH_SIZE = 16
EMBED_SIZE = 200

GLOVE_VECS_200D = torchtext.vocab.GloVe(name='6B', dim = EMBED_SIZE)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Helper functions

# Convert Unicode letter to Ascii
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters in a string
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

# Load SNLI dataset from torchtext
inputs = data.Field(lower = True, tokenize = 'spacy')
relations = data.Field(sequential = False)

train, dev, test = datasets.SNLI.splits(inputs, relations)

inputs.build_vocab(snli_train, snli_dev, snli_test)
relations.build_vocab(snli_train)

inputs.vocab.load_vectors(GLOVE_VECS_200D)

#print(GLOVE_VECS_200D["the"])

train_iter, dev_iter, test_iter = data.BucketIterator.splits((train, dev, test),
                                    batch_size = BATCH_SIZE, device = device)
