# Script for training and saving the generation models

import data
import gen_encoder
import gen_decoder

import argparse
import os
import pathlib
import time
import math

import torch
import torch.nn as nn
from torch import optim
import pandas as pd

LEARNING_RATE = 0.005
PAD_ID = 1
INIT_TOKEN_ID = data.inputs.vocab.stoi[data.INIT_TOKEN]

ABS_PATH = pathlib.Path(__file__).parent.absolute() 
MODELS_PATH = os.path.join(str(ABS_PATH), 'models/')

GENERATION_TYPES = ["entailment", "contradiction"]
MODEL_FNAMES = {
    "entailment":"entail-gen.tar",
    "contradiction":"contra-gen.tar"    
}

# Negative likelihood loss, with SGD
criterion = nn.NLLLoss()

# Function to train the seq2seq model, given a minibatch in the seq2seq model:
#   (1) Run input batch through encoder, producing final hidden + cell state
#   (2) Decoder takes in first word as initial input, and the final hidden and cell 
#       state from the encoder, as its initial hidden/cell state
#   (3) Use teacher forcing to train while processing decoder
#       At each timestep of the decoder, generate prob. vectors corresponding
#       to words of output lang. Then feed in the actual (target) tensor as 
#       the input token for the next timestep
#   (4) Once decoder has finished, compute loss between targets/generated 
#       sentences using NLL
#   (5) Perform backpropogation and stochastic gradient descent on both the
#       encoder and decoder networks
# Parameters:
#   batch - the batch from torchtext.data
#   encoder/decoder (optimizer) - the encoder/decoder (optimizer), respectively
def train_batch(batch, encoder, decoder, encoder_optimizer, decoder_optimizer, device):

    premises = batch.premise
    hypotheses = batch.hypothesis

    batch_size = batch.batch_size
    prem_seq_len = premises.size(1)

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
            
    loss = 0

    encoder_hidden = encoder.initHidden(batch_size, device)
    encoder_cell = encoder.initCell(batch_size, device)
    encoder_outputs = torch.zeros(premises.size(0), premises.size(1), 
                        encoder.hidden_size, device=device)

    batch_idxs = torch.arange(batch_size, dtype=torch.int64, device=device)
    seq_idxs = torch.arange(prem_seq_len, dtype=torch.int64, device=device)

    # Feed input through encoder, track encoder outputs for attention
    for i in range(premises.size(1)):
        # Only pass examples not yet finished processing
        not_padded = [j for j in range(premises.size(0)) if premises[j,i] != PAD_ID]
        not_padded_bool = torch.tensor([idx in not_padded for idx in batch_idxs], 
                            device=device)
        not_padded_bool = not_padded_bool.view(-1, 1).repeat(1,
            data.HIDDEN_SIZE).view(batch_size, data.HIDDEN_SIZE)

        encoder_input = premises[:, i:i+1]

#        curr_hidden = encoder_hidden[:,not_padded]
#        curr_cell = encoder_cell[:,not_padded]

        curr_hidden = torch.where(not_padded_bool, encoder_hidden[0], 
            torch.tensor(0., device=device)).unsqueeze(0)
        curr_cell = torch.where(not_padded_bool, encoder_cell[0],
            torch.tensor(0., device=device)).unsqueeze(0)

        encoder_out, (next_hidden, next_cell) = encoder(encoder_input,
            curr_hidden, curr_cell)

        # Update overall hidden/cell
        encoder_hidden = torch.where(not_padded_bool, next_hidden[0], encoder_hidden[0])
        encoder_cell = torch.where(not_padded_bool, next_cell[0], encoder_cell[0])

        encoder_hidden = encoder_hidden.unsqueeze(0)
        encoder_cell = encoder_cell.unsqueeze(0)

        not_padded_and_seq_bool = torch.tensor([[b_idx in not_padded and seq_idx == i
            for seq_idx in seq_idxs] for b_idx in batch_idxs], device=device)
        not_padded_and_seq_bool = not_padded_and_seq_bool.unsqueeze(2).repeat(1,1,
            data.HIDDEN_SIZE)

        encoder_outputs = torch.where(not_padded_and_seq_bool, 
            encoder_out[:,0:1].repeat(1,len(seq_idxs),1), encoder_outputs)

    # Decoder setup -> forward propogation
    decoder_input = torch.tensor([INIT_TOKEN_ID], device=device)
    decoder_input = decoder_input.repeat(batch_size)

    decoder_hidden = encoder_hidden[0]
    decoder_cell = encoder_cell[0]

    # Feed actual target token as input to next timestep
    for i in range(hypotheses.size(1)):
        # Only pass examples that are not done processing
        not_padded = [j for j in range(hypotheses.size(0)) if hypotheses[j,i] != PAD_ID]
        not_padded_bool = torch.tensor([idx in not_padded for idx in batch_idxs], 
                            device=device)
        not_padded_bool = not_padded_bool.view(-1, 1).repeat(1,
            data.HIDDEN_SIZE).view(batch_size, data.HIDDEN_SIZE)

        if i > 0:
            decoder_input = hypotheses[:, i]
        else:
            decoder_input = decoder_input.unsqueeze(1)        

        curr_hidden = torch.where(not_padded_bool, decoder_hidden,
            torch.tensor(0., device=device))
        curr_cell = torch.where(not_padded_bool, decoder_cell,
            torch.tensor(0., device=device))

        decoder_output, next_hidden, next_cell, decoder_attn = decoder(decoder_input,
            curr_hidden, curr_cell, encoder_outputs, not_padded, device)

#        decoder_hidden[not_padded] = next_hidden
#        decoder_cell[not_padded] = next_cell

        decoder_hidden = torch.where(not_padded_bool, next_hidden, decoder_hidden)
        decoder_cell = torch.where(not_padded_bool, next_cell, decoder_cell)

        # Compute loss
        loss += criterion(decoder_output, hypotheses[:,i])
            
    # Backpropogation + Gradient descent
    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    # Return the total loss for the batch
    return loss.item()


# Train for the specified number of epochs
def trainIterations(encoder, decoder, train_iter, n_epochs, device, print_every = 10):
    start = time.time()
    print_loss_total = 0

    train_iter.init_epoch()
    
    encoder.train()
    decoder.train()

    encoder_optimizer = optim.SGD(encoder.parameters(), lr = LEARNING_RATE, weight_decay = 0.01)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr = LEARNING_RATE, weight_decay = 0.01)

    # Process the training set in batches, for NUM_EPOCHS epochs
    for epoch in range(n_epochs):
        print_loss_total = 0

        print("Epoch: ", epoch)

        train_iter.init_epoch()

        # Process each batch and perform optimization accordingly
        for batch_num, batch in enumerate(train_iter):
            # Train and retrieve total loss for the batch
            loss = train_batch(batch, encoder, decoder, encoder_optimizer,
                decoder_optimizer, device)

            print_loss_total += loss        
            
#            if((batch_num + 1) % print_every == 0):
            if (1 == 1) :
                print("batch num:", batch_num)
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print("avg loss:", print_loss_avg)

            if batch_num == 15:
                break


        print("-------------------------------------------------------------\n")

# Main
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, help="output directory")
    parser.add_argument("--model_type", type=str, help="entailment or contradiction",
                        choices = GENERATION_TYPES)
    parser.add_argument("--num_epochs", type=int, default=1)                        

    args = parser.parse_args()

    torch.manual_seed(321)

    # Create encoder/decoder
    encoder = gen_encoder.Encoder(len(data.inputs.vocab), data.HIDDEN_SIZE,
        embeddings = data.inputs.vocab.vectors, embed_size = data.EMBED_SIZE)
    decoder = gen_decoder.Decoder(len(data.inputs.vocab), data.HIDDEN_SIZE, 
        embeddings = data.inputs.vocab.vectors, embed_size = data.EMBED_SIZE)

    print("Starting Training.")
    encoder.train()
    decoder.train()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder.to(device) 
    decoder.to(device)

    # Train the models on the filtered dataset
    trainIterations(encoder, decoder, data.TRAIN_ITER_DICT[args.model_type], 
                    args.num_epochs, device)

    print("Training Complete.")

    # Save models
    if not os.path.isdir(MODELS_PATH):
        print('creating directory %s ' % MODELS_PATH)
        os.mkdir(MODELS_PATH)

    print("Saving models...")
    torch.save({
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),            
        }, 
        os.path.join(ABS_PATH, args.output_dir, MODEL_FNAMES[args.model_type])
    )

if __name__ == "__main__":
    main()
