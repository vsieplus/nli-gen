# Script for preparing data for generation model

import torch
import torch.nn as nn
from torch import optim

import torchtext
from torchtext import data, datasets

import numpy as np
import unicodedata

import lang
import re
import random

EMBED_SIZE = 200

GLOVE_VECS_200D = torchtext.vocab.GloVe(name='840B', dim = EMBED_SIZE)

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

snli_train, snli_dev, snli_test = datasets.SNLI.splits(inputs, relations)

inputs.build_vocab(snli_train, snli_dev, snli_test)
relations.build_vocab(snli_train)

inputs.vocab.load_vectors(GLOVE_VECS_200D)
