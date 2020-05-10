# The Decoder RNN for the generation model, usable both with/without attention

import torch
import torch.nn as nn

import seq_helper

# Define a class for the decoder, using the LSTM recurrent unit and word-word attention
class Decoder(nn.Module):

    # Constructor:
    #   output_size - # of features of the output (vocab size)
    #   hidden_size - # of features of the hidden state vectors
    #   embed_size - # of features in embedding vectors
    #   embeddings - (optional pretrained embeddings)
    def __init__(self, vocab_size, hidden_size, embed_size, embeddings = None):
        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.num_lstm_layers = 1

        # Embeddings for our input tokens (which are previous outputs) at each timestep
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.embedding.weight.data.copy_(embeddings)

        # Attention mlp
        #   - Linear layer: take hidden state at previous timestep, and a
        #     context (hidden state) for a timestep i (from the encoder)
        #   - outputs a weight corresponding to timestep i
        self.attn = nn.Linear(hidden_size * 2, 1)

        self.softmax = nn.Softmax(dim = -1)

        # Use the LSTM cell as recurrent unit
        self.lstm_cell = nn.LSTMCell(embed_size + hidden_size, hidden_size)
    
        # Generate output using linear transform of hidden state from LSTM -> logsoftmax
        self.linear_out = nn.Linear(hidden_size, vocab_size)
        self.log_softmax = nn.LogSoftmax(dim = -1)

    # Forward propogation through decoder (step-by-step)
    #   input_batch - tensor for the input token(s) for current timestep with shape: (batch_size, 1),
    #                 containing corresponding indices of words of the sentences
    #   prev_h - previous hidden state, shape: (batch, hidden_size)
    #   prev_c - previous cell state, shape: (same as hidden)
    #   encoder_outputs - outputs of encoder RNN, shape: (batch_size, seq_len, hidden_size)
    #   curr_idxs -> current indices being processed
    def forward(self, input_batch, prev_h, prev_c, encoder_outputs, curr_idxs, device):  
        # Get embedding of the input
        input_embeddings = self.embedding(input_batch)
        
        seq_len = encoder_outputs.size(1)

        # Compute attention weights - shape: (batch, seq_len)
        # For each timestep i, compute result of feeding concatenation 
        # [prev_h, h_encoder_i] through the attention MLP;
        attn_weights = torch.zeros(encoder_outputs.size(0), seq_len, device=device)

        for index in curr_idxs:
            for j in range(seq_len):
                attn_weights[index, j] = self.attn(torch.cat((prev_h[index], 
                    encoder_outputs[index, j]), dim = -1))

#       for j in range(seq_len):
#           attn_weights = torch.where(seq_len == j, self.attn(torch.cat((prev_h,
#               encoder_outputs[:,j]), dim = -1)), attn_weights)

        # Normalize the attention weights using a softmax layer; shape: (batch, seq_len)
        attn_weights = self.softmax(attn_weights)

        # Compute context vectors by summing product of attention weights with
        # outputs of encoder RNN (weighted sum), using batch mat-mul
        #   (b x 1 x seq) * (b x seq x hidden) -> (b x 1 x hidden) [context
        #   vectors for each sent. in the batch]
        context_vecs = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)

        # Concatentate embeddings with context vectors

        input_attended = torch.cat((input_embeddings.squeeze(1), 
                                    context_vecs.squeeze(1)), dim = -1)

        # Feed to LSTM along with previous hidden/cell state
        hidden, cell = self.lstm_cell(input_attended, (prev_h, prev_c))
        output = self.linear_out(hidden)
        output = self.log_softmax(output)
        return output, hidden, cell, attn_weights
