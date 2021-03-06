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
import torch.nn.utils.rnn as rnn_utils
from torch import optim
import pandas as pd

LEARNING_RATE = 0.01

ABS_PATH = pathlib.Path(__file__).parent.absolute() 
MODELS_PATH = os.path.join(str(ABS_PATH), 'models/')

GENERATION_TYPES = ["entailment", "contradiction"]
MODEL_FNAMES = {
    "entailment":"entail-gen.tar",
    "contradiction":"contra-gen.tar"    
}

# Negative likelihood loss, with SGD
criterion = nn.NLLLoss()

def get_sent_lengths(batch, device):
    pad_token_indices = [(ex == data.PAD_TOKEN_ID).nonzero() for ex in batch]

    batch_lengths = []
    for pad_tokens in pad_token_indices:
        if pad_tokens.size(0) == 0:
            batch_lengths.append(batch.size(1))        # no pad/max length
        else:
            batch_lengths.append(pad_tokens[0].item()) # first pad idx

    return torch.tensor(batch_lengths, device = device)


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

    encoder.train()
    decoder.train()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    premises = batch.premise
    hypotheses = batch.hypothesis

    batch_size = batch.batch_size
            
    loss = 0

    premise_lengths = get_sent_lengths(premises, device)
    hypothesis_lengths = get_sent_lengths(hypotheses, device)

    # Feed input through encoder, store outputs
    encoder_out, (encoder_hidden, encoder_cell) = encoder(premises, premise_lengths,
        encoder.initHidden(batch_size, device), encoder.initCell(batch_size, device))

    # Decoder setup -> forward propogation
    decoder_out, decoder_hidden, decoder_cell = decoder(hypotheses, hypothesis_lengths,
        encoder_hidden, encoder_cell, device)

    # compute loss for each timestep (except the last)
    for i in range(hypotheses.size(1) - 1):
        target = hypotheses[:,i+1] 

        # Compute loss
        loss += criterion(decoder_out[:,i], target)
            
    # Backpropogation + Gradient descent
    loss.backward()

    decoder_optimizer.step()
    encoder_optimizer.step()

    # Return the avg loss for the batch
    return loss.item() / batch_size

def evaluate(dev_iter, encoder, decoder, device):
    encoder.eval()
    decoder.eval()

    total_loss = 0

    with torch.no_grad():
        dev_iter.init_epoch()

        for batch in dev_iter:
            loss = 0
            batch_size = batch.batch_size

            premises = batch.premise.to(device)
            hypotheses = batch.hypothesis.to(device)

            premise_lengths = get_sent_lengths(premises, device)
            hypothesis_lengths = get_sent_lengths(hypotheses, device)

            encoder_out, (encoder_hidden, encoder_cell) = encoder(premises, premise_lengths,
                encoder.initHidden(batch_size, device), encoder.initCell(batch_size, device))

            decoder_out, decoder_hidden, decoder_cell = decoder(batch.hypothesis,
                hypothesis_lengths, encoder_hidden, encoder_cell, device)

            for i in range(batch.hypothesis.size(1) - 1):
                loss += criterion(decoder_out[:,i], hypotheses[:,i+1])

            total_loss += loss.item()

    return total_loss / len(dev_iter)

# Train for the specified number of epochs
def trainIterations(encoder, decoder, train_iter, dev_iter, n_epochs, device, args, print_every = 200):
    start = time.time()
    print_loss_total = 0

    encoder.train()
    decoder.train()

    encoder_optimizer = optim.Adam(encoder.parameters(), lr = LEARNING_RATE)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr = LEARNING_RATE)

    min_valid_loss = float('inf')

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
            
            if((batch_num + 1) % print_every == 0):
                print("batch num:", batch_num + 1)
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print("avg loss:", print_loss_avg)

        # check for early stopping every other epoch
        if epoch % 2 == 0:
            valid_loss = evaluate(dev_iter, encoder, decoder, device)
            print("validation loss: %.3f" % valid_loss)

            if valid_loss < min_valid_loss:
                min_valid_loss = valid_loss
                save_model(args, encoder, decoder)
            else:
                print("stopping early...")
                break


        print("-------------------------------------------------------------\n")

def save_model(args, encoder, decoder):
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

# Main
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, help="output directory")
    parser.add_argument("--model_type", type=str, help="entailment or contradiction",
                        choices = GENERATION_TYPES)
    parser.add_argument("--num_epochs", type=int, default=6)

    args = parser.parse_args()

    torch.manual_seed(321)

    # Create encoder/decoder
    encoder = gen_encoder.Encoder(len(data.inputs.vocab), data.HIDDEN_SIZE,
        embeddings = data.inputs.vocab.vectors, embed_size = data.EMBED_SIZE, 
        pad_idx = data.PAD_TOKEN_ID, unk_idx = data.UNK_TOKEN_ID)
    decoder = gen_decoder.Decoder(len(data.inputs.vocab), data.HIDDEN_SIZE, 
        embeddings = data.inputs.vocab.vectors, embed_size = data.EMBED_SIZE,
        pad_idx = data.PAD_TOKEN_ID, unk_idx = data.UNK_TOKEN_ID)


    print("Starting Training.")
    encoder.train()
    decoder.train()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder.to(device) 
    decoder.to(device)

    # Train the models on the filtered dataset
    trainIterations(encoder, decoder, data.TRAIN_ITER_DICT[args.model_type], 
        data.DEV_ITER_DICT[args.model_type], args.num_epochs, device, args)

    print("Training Complete.")
    

if __name__ == "__main__":
    main()
