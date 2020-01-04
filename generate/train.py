# Script for training the generation model, and providing user functionality
# to interact with it afterwards

import preprocessing as pp
import gen_encoder
import gen_decoder

import time
import math

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
#   batch_input_tensor - [packed] tensor of shape (sent_len, batch_size, vocab_size)
#                        representing a batch of input sentences as one-hot vectors
#   batch_target_tensor - tensor of shape (batch_size, sent_len, vocab_size)
#                         representing a batch of target sentences as one-hot_vectors
#   encoder/decoder (optimizer) - the encoder/decoder (optimizer), respectively
def train_batch(batch_input_tensor, batch_target_tensor, encoder, decoder,
    encoder_optimizer, decoder_optimizer):

    curr_batch_size = batch_target_tensor.size(0)

    encoder_hidden = encoder.initHidden(curr_batch_size)

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    target_length = batch_target_tensor.size(1)
            
    loss = 0

    # Feed input through encoder, store packed output + context
    encoder_outputs, (hn, cn) = encoder(batch_input_tensor, 
        encoder.initHidden(curr_batch_size),
        encoder.initCell(curr_batch_size), curr_batch_size)

    # Unpack outputs
    encoder_outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(encoder_outputs) 

    # If outputs not of length MAX_SENT_LEN, pad with zeros
    if(encoder_outputs.size(0) != processing.MAX_SENT_LEN):
        encoder_outputs = torch.cat((encoder_outputs, torch.zeros(
            processing.MAX_SENT_LEN - encoder_outputs.size(0), 
            curr_batch_size, hidden_size)), 0)

    # Decoder setup -> forward propogation
    decoder_input = torch.tensor([lang.SOS_token], dtype = torch.long).view(-1,1)
    decoder_input = torch.tensor([decoder_input] * curr_batch_size)

    context = (hn, cn)
    decoder_hidden = context[0].squeeze()
    decoder_cell = context[1].squeeze()

    # Feed actual target token as input to next timestep
    for timestep in range(min(processing.MAX_SENT_LEN, target_length)):
        decoder_output, decoder_hidden, decoder_cell, decoder_attn = decoder(
            decoder_input, decoder_hidden, decoder_cell, encoder_outputs)

        continue_idxs = []
        loss_idxs = []

        # Compute loss for data in batch that have not passed <EOS> yet,
        # and only continue computations 
        for b in range(curr_batch_size):
            curr_target_idx = batch_target_tensor[b, timestep][0]

            # If current target index > 0, compute loss, and update states
            if curr_target_idx > 0:
                loss_idxs.append(b)

                # If not at end of sentence, continue computation for this ex
                if curr_target_idx > lang.EOS_token:
                    continue_idxs.append(b)

        loss += loss_F(decoder_output[loss_idxs,:], batch_target_tensor[loss_idxs, timestep].squeeze(1))

        decoder_input = batch_target_tensor[:, timestep].squeeze(1)
            
    # Backpropogation + Gradient descent
    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    # Return the average loss
    return loss.item()/target_length


# Train for the specified number of epochs
def trainIterations(encoder, decoder, n_epochs, print_every = 1000):
    start = time.time()
    print_loss_total = 0

    encoder_optimizer = optim.SGD(encoder.parameters(), lr = learning_rate, weight_decay = 0.01)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr = learning_rate, weight_decay = 0.01)
    
    # Process the training set in batches, n_iters times
    for epoch in range(pp.NUM_EPOCHS):
        print_loss_total = 0

        print("Epoch: ", epoch)
        for batch_num in range(math.ceil(len(training_pairs)/batch_size) - 1):
            batch = training_pairs[(batch_num * batch_size):(batch_num * batch_size) + batch_size]
            batch_input_tensor, batch_target_tensor = processing.batchToTensor(batch) 

            loss = train_batch(batch_input_tensor, batch_target_tensor, encoder, decoder,
                encoder_optimizer, decoder_optimizer)

            print_loss_total += loss

            if((batch_num + 1) * batch_size % print_every == 0):
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %.2f%%) [Avg. loss]: %.4f' % (timeSince(start, 
                        ((batch_num + 1) * batch_size / len(training_pairs))), 
                    (batch_num + 1) * batch_size, 
                    (batch_num + 1) * batch_size / len(training_pairs) * 100,
                    print_loss_avg))
        print("-------------------------------------------------------------\n")


# Create encoder/decoder
encoder1 = gen_encoder.Encoder(pp.iputs.vocab.max_size, pp.HIDDEN_SIZE,
    embeddings = pp.GLOVE_VECS_200D)
decoder1 = gen_decoder.Decoder(pp.inputs.vocab.max_size, pp.HIDDEN_SIZE, 
    embeddings = pp.GLOVE_VECS_200D)

# Train the models
trainIterations(encoder1, decoder1, n_epochs = pp.NUM_EPOCHS)

print("-----------------Training Complete-------------------------------------\n")

############### Evaluation/Testing on the test set ############################

print("----------------- Evaluation -------------------------------------")
