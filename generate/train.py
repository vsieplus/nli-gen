# Script for training the generation model, and providing user functionality
# to interact with it afterwards

import preprocessing as pp
import gen_encoder
import gen_decoder

import time
import math

import pandas as pd

###############################################################################
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))
###############################################################################

GENERATION_TYPES = ["entailment", "contradiction"]

torch.manual_seed(321)

# Negative likelihood loss, with SGD
loss_F = nn.NLLLoss()

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
def train_batch(batch, encoder, decoder, encoder_optimizer, decoder_optimizer):

    curr_batch_size = batch.batch_size
    batch.premise = torch.tensor(batch.premise, dtype = torch.long, device = device)
    batch.hypothesis = torch.tensor(batch.hypothesis, dtype = torch.long, device = device)

    encoder_hidden = encoder.initHidden(curr_batch_size)

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
            
    loss = 0

    # Feed input through encoder, store packed output + context
    encoder_outputs, (hn, cn) = encoder(batch.premise,
        encoder.initHidden(curr_batch_size), encoder.initCell(curr_batch_size), 
        curr_batch_size)

#    # If outputs not of length MAX_SENT_LEN, pad with zeros
#    if(encoder_outputs.size(0) != processing.MAX_SENT_LEN):
#        encoder_outputs = torch.cat((encoder_outputs, torch.zeros(
#            processing.MAX_SENT_LEN - encoder_outputs.size(0), 
#            curr_batch_size, hidden_size)), 0)

    # Decoder setup -> forward propogation
    decoder_input = torch.tensor(pp.inputs.init_token, dtype = torch.long).view(-1,1)
    decoder_input = torch.tensor([decoder_input] * curr_batch_size)

    context = (hn, cn)
    decoder_hidden = context[0].squeeze()
    decoder_cell = context[1].squeeze()

    # Feed actual target token as input to next timestep
    for timestep in range(min(processing.MAX_SENT_LEN, target_length)):
        decoder_output, decoder_hidden, decoder_cell, decoder_attn = decoder(
            decoder_input, decoder_hidden, decoder_cell, 
            encoder_outputs[continue_idxs, :])

        loss_idxs = []
        continue_idxs = []

        # Compute loss for data in batch that have not passed <EOS> yet,
        # and only continue computations 
        for b in range(curr_batch_size):
            curr_target_idx = batch.hypothesis[b, timestep][0]

            # If current target index > 0, compute loss, and update states
            if curr_target_idx > 0:
                loss_idxs.append(b)

                if curr_target_idx != pp.inputs.eos_token:
                    continue_idxs.append(b)

        loss += loss_F(decoder_output[loss_idxs,:], batch.hypothesis[loss_idxs, timestep].squeeze(1))

        decoder_input = batch.hypothesis[continue_idxs, timestep].squeeze(1)
            
    # Backpropogation + Gradient descent
    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    # Return the total loss for the batch
    return loss.item()


# Train for the specified number of epochs
def trainIterations(encoder, decoder, train_iter, n_epochs, print_every = 1000):
    start = time.time()
    print_loss_total = 0

    train_iter.init_epoch()
    
    encoder.train()
    decoder.train()

    encoder_optimizer = optim.SGD(encoder.parameters(), lr = learning_rate, weight_decay = 0.01)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr = learning_rate, weight_decay = 0.01)

    # Process the training set in batches, for NUM_EPOCHS epochs
    for epoch in range(pp.NUM_EPOCHS):
        print_loss_total = 0

        print("Epoch: ", epoch)

        # Process each batch and perform optimization accordingly
        for batch_num, batch in enumerate(train_iter):
            # Train and retrieve total loss for the batch
            loss = train_batch(batch, encoder, decoder, encoder_optimizer,
                decoder_optimizer)

            print_loss_total += loss        
            
            if((batch_num + 1) * train_iter.size % print_every == 0):
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %.2f%%) [Avg. loss]: %.4f' % (timeSince(start, 
                        ((batch_num + 1) * train_iter.batch_size / len(training_pairs))), 
                         (batch_num + 1) * train_iter.batch_size,
                         (batch_num + 1) * train_iter.batch_size / len(training_pairs) * 100,
                         print_loss_avg))

        print("-------------------------------------------------------------\n")


# Create encoder/decoder
entail_encoder = gen_encoder.Encoder(pp.iputs.vocab.max_size, pp.HIDDEN_SIZE,
    embeddings = pp.GLOVE_VECS_200D)
entail_decoder = gen_decoder.Decoder(pp.inputs.vocab.max_size, pp.HIDDEN_SIZE, 
    embeddings = pp.GLOVE_VECS_200D)

contradict_encoder = gen_encoder.Encoder(pp.iputs.vocab.max_size, pp.HIDDEN_SIZE,
    embeddings = pp.GLOVE_VECS_200D)
contradict_decoder = gen_decoder.Decoder(pp.inputs.vocab.max_size, pp.HIDDEN_SIZE, 
    embeddings = pp.GLOVE_VECS_200D)

# Train the models for entailment and contradiction
trainIterations(entail_encoder, entail_decoder, pp.train_iter_entail, n_epochs = pp.NUM_EPOCHS)
trainIterations(contradict_encoder, contradict_decoder, pp.train_iter_contradict, n_epochs = pp.NUM_EPOCHS)

print("-----------------Training Complete-------------------------------------\n")

############### Evaluation/Testing on the test set ############################

print("----------------- Evaluation -------------------------------------")

entail_encoder.eval()
entail_decoder.eval()
contradict_encoder.eval()
contradict_decoder.eval()

# Function to perform evaluation/write out results of model for a given batch
def test_batch(batch, encoder, decoder, df):
    curr_batch_size = batch.batch_size
    result_dicts = [{} for _ in range(curr_batch_size)]

    for b in range(curr_batch_size):
        result_dicts[b]["premise"] = ""
        for w in range(batch.premise[b, :].size(1)):
            word_idx = batch.premise[b,w]
            result_dicts[b]["premise"] += pp.inputs.iots[word_idx] + " "

    # Feed input through encoder, store packed output + context
    encoder_outputs, (hn, cn) = encoder(batch.premise,
        encoder.initHidden(curr_batch_size), encoder.initCell(curr_batch_size), 
        curr_batch_size)

    # Decoder setup -> forward propogation
    decoder_input = torch.tensor(pp.inputs.init_token, dtype = torch.long).view(-1,1)
    decoder_input = torch.tensor([decoder_input] * curr_batch_size)

    context = (hn, cn)
    decoder_hidden = context[0].squeeze()
    decoder_cell = context[1].squeeze()

    # Feed actual target token as input to next timestep
    for timestep in range(min(processing.MAX_SENT_LEN, target_length)):
        decoder_output, decoder_hidden, decoder_cell, decoder_attn = decoder(
            decoder_input, decoder_hidden, decoder_cell, 
            encoder_outputs[continue_idxs, :])

        continue_idxs = []

        # Continue/print generation for data in batch that have not passed <EOS> yet,
        for b in range(curr_batch_size):
            curr_target_idx = batch.hypothesis[b, timestep][0]

            # If current target index 
            if curr_target_idx != 0:
                continue_idxs.append(b)

        # Input to next timestep are argmax indices of decoder output
        decoder_input = torch.argmax(decoder_output[continue_idxs, :], dim = -1)

        # Detokenize (to text) and write results to file
        for b in range(curr_batch_size):
            result_dicts[b]["hypothesis"] = ""
            if b in continue_idx:
                word_b = pp.inputs.iots[decoder_input[b,0]]        
                result_dicts[b]["hypothesis"] += word_b + " "

    rows_list.extend(result_dicts)

rows_list_entail = []
for batch_num, batch in enumerate(pp.test_iter_entail):
    test_batch(batch, rows_list_entail)

rows_list_contradict = []
for batch_num, batch in enumerate(pp.test_iter_contradict):
    test_batch(batch, rows_list_contradict)

df_entail = pd.DataFrame(rows_list_entail, columns = ("premise", "hypothesis"))
df_contradict = pd.DataFrame(rows_list_contradict, columns = ("premise", "hypothesis"))
df_entail.to_csv("model_test_outputs_entailments.tsv", sep = "\t")
df_contradict.to_csv("model_test_outputs_contradictions.tsv", sep = "\t")
