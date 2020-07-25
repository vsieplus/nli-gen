# The Encoder RNN for the generation model, using LSTM

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Define class for the encoder, using the LSTM recurrent unit
class Encoder(nn.Module):

    # Constructor:
    #   vocab_size - size of vocabulary
    #   hidden_size - dimension of hidden states
    #   embed_size - dimension of embeddings
    #   embeddings - (pretrained embeddings)
    def __init__(self, vocab_size, hidden_size, embed_size, pad_idx, unk_idx, embeddings = None):
        super(Encoder, self).__init__()

        self.hidden_size = hidden_size
        
        # Embeddings for our input tokens at each timestep
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx = pad_idx)
        self.embedding.weight.data.copy_(embeddings)

        self.embedding.weight.data[pad_idx] = torch.zeros(embed_size)
        self.embedding.weight.data[unk_idx] = torch.zeros(embed_size)

        # LSTM RNN, accepts:
        #   - input of shape (batch_size, seq_len) [embeddings]
        #   - initial hidden and cell states 
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first = True)

    # Forward propogation through the Encoder
    #   input - the input batch (packed), with shape: (batch_size, seq_len),
    #           containing corresponding indices of words of the sentences
    #   h - hidden state, shape: (1, batch, hidden_size)
    #   c - cell state, shape: (same as h0)
    def forward(self, input_batch, input_lengths, h0, c0):
        # Perform embedding element wise
        input_batch = self.embedding(input_batch)

        input_packed = pack_padded_sequence(input_batch, input_lengths, batch_first = True,
            enforce_sorted = False)

        # Pass through lstm
        output, (hidden, cell) = self.lstm(input_packed, (h0, c0))

        output, _ = pad_packed_sequence(output, batch_first = True)

        return output, (hidden, cell)

    # Initial states for the LSTM
    def initHidden(self, batch_size, device):
        return torch.zeros(1, batch_size, self.hidden_size, device = device)

    def initCell(self, batch_size, device):
        return torch.zeros(1, batch_size, self.hidden_size, device = device)
