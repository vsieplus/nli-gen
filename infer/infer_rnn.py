# The inference rnn model

import torch
import torch.nn as nn

class InferRNN(nn.Module):
    # Constructor:
    #   vocab_size - size of vocabulary
    #   hidden_size - dimension of hidden states
    #   embed_size - dimension of embeddings
    #   embeddings - (pretrained embeddings)
    def __init__(self, vocab_size, hidden_size, embed_size, embeddings = None,
        num_classes = 3):
        super(InferRNN, self).__init__()

        self.hidden_size = hidden_size
        
        # Embeddings for our input tokens at each timestep
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.embedding.weight.data.copy_(embeddings)

        # LSTM RNN, accepts:
        #   - input of shape (batch_size, seq_len) [embeddings]
        #   - initial hidden and cell states 
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first = True)

        # linear output for last layer, output -> mlp
        # last layer has 3 nodes -> possible outputs (entailment, contradiction, neutral)
        self.linear_out = nn.Linear(hidden_size, num_classes)

        # end softmax
        self.log_softmax = nn.LogSoftmax(dim = -1)

    # Forward propogation through the rnn + mlp
    #   input - the input batch (packed), with shape: (batch_size, seq_len),
    #           containing corresponding indices of words of the sentences
    #   Returns vector of 3 probs after log-softmax, 1 for each class
    def forward(self, input_batch, device):
        # Perform embedding element wise
        input_batch = self.embedding(input_batch)

        batch_size = input_batch.size(0)

        h0 = self.initHidden(batch_size, device)
        c0 = self.initCell(batch_size, device)

        # Pass through lstm
        output, (hidden, cell) = self.lstm(input_batch, (h0, c0))

        linear_output = self.linear_out(output)
        norm_output = self.log_softmax(linear_output)

        return norm_output

    # Initial states for the LSTM
    def initHidden(self, batch_size, device):
        return torch.zeros(1, batch_size, self.hidden_size, device = device)

    def initCell(self, batch_size, device):
        return torch.zeros(1, batch_size, self.hidden_size, device = device)