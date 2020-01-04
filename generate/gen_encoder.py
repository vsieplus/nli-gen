# The Encoder RNN for the generation model, using LSTM

import torch
import torch.nn as nn

import seq_helper

# Define class for the encoder, using the LSTM recurrent unit
class Encoder(nn.Module):

    # Constructor:
    #   vocab_size - size of vocabulary
    #   hidden_size - dimension of hidden states
    #   embed_size - dimension of embeddings
    #   embeddings - (optional pretrained embeddings)
    def __init__(self, vocab_size, hidden_size, embed_size, embeddings = None):
        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        
        # Embeddings for our input tokens at each timestep
        if embeddings is None:
            # If no embeddings provided, use random ones
            self.embedding = nn.Embedding(vocab_size, embed_size)
        else:
            self.embedding = nn.Embedding.from_pretrained(embeddings)

        # LSTM RNN, accepts:
        #   - input of shape (batch_size, seq_len) [embeddings]
        #   - initial hidden and cell states 
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first = True)

    # Forward propogation through the Encoder
    #   input - the input batch (packed), with shape: (batch_size, seq_len),
    #           containing corresponding indices of words of the sentences
    #   h0 - initial hidden state, shape: (1, batch, hidden_size)
    #   c0 - initial cell state, shape: (same as h0)
    #   batch_size - size of batch fed into network
    def forward(self, input_batch, h0, c0, batch_size):
        # Perform embedding element wise on packed sequence
        input_batch = seq_helper.packed_f(self.embedding, input_batch)

        # Pass through lstm
        output, (hidden, cell) = self.lstm(input_batch, (h0, c0))
        return output, (hidden, cell)


    # Initial states for the LSTM
    def initHidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)

    def initCell(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)
