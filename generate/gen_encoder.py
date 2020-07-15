# The Encoder RNN for the generation model, using LSTM

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils

# Define class for the encoder, using the LSTM recurrent unit
class Encoder(nn.Module):

    # Constructor:
    #   vocab_size - size of vocabulary
    #   hidden_size - dimension of hidden states
    #   embed_size - dimension of embeddings
    #   embeddings - (pretrained embeddings)
    def __init__(self, vocab_size, hidden_size, embed_size, embeddings = None):
        super(Encoder, self).__init__()

        self.hidden_size = hidden_size
        
        # Embeddings for our input tokens at each timestep
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.embedding.weight.data.copy_(embeddings)

        # LSTM RNN, accepts:
        #   - input of shape (batch_size, seq_len) [embeddings]
        #   - initial hidden and cell states 
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first = True)

    # Forward propogation through the Encoder
    #   input - the input batch (packed), with shape: (batch_size, seq_len),
    #           containing corresponding indices of words of the sentences
    #   h - hidden state, shape: (1, batch, hidden_size)
    #   c - cell state, shape: (same as h0)
    def forward(self, input_batch, h0, c0):
        # Perform embedding element wise
        input_batch = self.embedding(input_batch)

        # Pass through lstm
        output, (hidden, cell) = self.lstm(input_batch, (h0, c0))

        hidden = hidden[0]
        cell = cell[0]

        hidden = hidden.unsqueeze(0)
        cell = cell.unsqueeze(0)

        return output, (hidden, cell)

    # Initial states for the LSTM
    def initHidden(self, batch_size, device):
        return torch.zeros(1, batch_size, self.hidden_size, device = device)

    def initCell(self, batch_size, device):
        return torch.zeros(1, batch_size, self.hidden_size, device = device)
