# Helper functions for both encoder/decoder RNNs

import torch

# Apply a given function element wise to a packed sequence;
# Returns a new packed sequence with the resulting values
def packed_f(fn, packed_seq):
    return torch.nn.utils.rnn.PackedSequence(fn(packed_seq.data.squeeze()), packed_seq.batch_sizes)
