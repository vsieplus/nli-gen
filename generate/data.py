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
NUM_EPOCHS = 25
HIDDEN_SIZE = 256
BATCH_SIZE = 1
EMBED_SIZE = 200

INIT_TOKEN = "<sos>"
GLOVE_VECS_200D = torchtext.vocab.GloVe(name='6B', dim = EMBED_SIZE)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load SNLI dataset from torchtext
inputs = data.Field(lower = True, tokenize = 'spacy', batch_first = True, 
                    init_token = INIT_TOKEN)
labels = data.Field(sequential = False, batch_first = True)

train, dev, test = datasets.SNLI.splits(inputs, labels)

train_entail = data.Dataset(train.examples, train.fields, 
    filter_pred = lambda x: x.label == "entailment")
dev_entail = data.Dataset(dev.examples, dev.fields,
    filter_pred = lambda x: x.label == "entailment")
test_entail = data.Dataset(test.examples, test.fields, 
    filter_pred = lambda x: x.label == "entailment")

train_contradict = data.Dataset(train.examples, train.fields, 
    filter_pred = lambda x: x.label == "contradiction")
dev_contradict = data.Dataset(dev.examples, dev.fields,
    filter_pred = lambda x: x.label == "contradict")
test_contradict = data.Dataset(test.examples, test.fields, 
    filter_pred = lambda x: x.label == "contradiction")

inputs.build_vocab(train_entail, test_entail, train_contradict, test_contradict)
labels.build_vocab(train_entail)

print("Loading embeddings...")

inputs.vocab.load_vectors(GLOVE_VECS_200D)

#print(GLOVE_VECS_200D["the"])

print("Setting up training and test sets")
train_iter, dev_iter, test_iter = data.BucketIterator.splits((train, dev, test),
                                    batch_size = BATCH_SIZE, device = device)
train_iter_entail, dev_iter_entail, test_iter_entail = data.BucketIterator.splits(
    (train_entail, dev_entail, test_entail), batch_size = BATCH_SIZE, device = device, sort = False)
#    sort_key=lambda x: len(x.premise), sort_within_batch = False)
train_iter_contradict, dev_iter_contradict, test_iter_contradict = data.BucketIterator.splits(
    (train_contradict, dev_contracit, test_contradict), batch_size = BATCH_SIZE, device = device, sort = False)
#    sort_key=lambda x: len(x.premise), sort_within_batch = False)

# Dicts for data iterators
TRAIN_ITER_DICT = {
    "entailment": train_iter_entail,
    "contradiction": train_iter_contradict,    
}
DEV_ITER_DICT = {
    "entailment": dev_iter_entail,
    "contradiction": dev_iter_contradict,    
}
TEST_ITER_DICT = {
    "entailment": test_iter_entail,
    "contradiction": test_iter_contradict,    
}
