# The Decoder RNN for the generation model, usable both with/without attention

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils

# Define a class for the decoder, using the LSTM recurrent unit and word-word attention
class Decoder(nn.Module):

    # Constructor:
    #   output_size - # of features of the output (vocab size)
    #   hidden_size - # of features of the hidden state vectors
    #   embed_size - # of features in embedding vectors
    #   embeddings - (optional pretrained embeddings)
    def __init__(self, vocab_size, hidden_size, embed_size, pad_idx, unk_idx, embeddings = None):
        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.num_lstm_layers = 1

        # Embeddings for our input tokens (which are previous outputs) at each timestep
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx = pad_idx)
        self.embedding.weight.data.copy_(embeddings)

        self.embedding.weight.data[pad_idx] = torch.zeros(embed_size)
        self.embedding.weight.data[unk_idx] = torch.zeros(embed_size)

        # Use the LSTM cell as recurrent unit
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first = True)
    
        # Linear layer to vocab
        self.linear_out = nn.Linear(hidden_size, vocab_size)

        # Generate output using logits from LSTM -> logsoftmax
        self.log_softmax = nn.LogSoftmax(dim = -1)

    # Forward propogation through decoder (step-by-step)
    #   input_batch - tensor for the input token(s) for current timestep with shape: (batch_size, 1),
    #                 containing corresponding indices of words of the sentences
    #   prev_h - previous hidden state, shape: (batch, hidden_size)
    #   prev_c - previous cell state, shape: (same as hidden)
    def forward(self, input_batch, prev_h, prev_c, device):  
        # Get embedding of the input
        input_embeddings = self.embedding(input_batch)

        # Feed to LSTM along with previous hidden/cell state
        output, (hidden, cell) = self.lstm(input_embeddings, (prev_h, prev_c))

        output = self.linear_out(output)
        output = self.log_softmax(output)
        output = output.squeeze(0)
        return output, hidden, cell
