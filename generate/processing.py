# Script for preparing data for generation model

import torch

from torchtext import data
from torchtext import datasets

import numpy as np
import unicodedata

import load_glove
import lang
import re
import random

EMBED_SIZE = 200

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
