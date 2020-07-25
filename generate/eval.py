# Evaluation script for the generation model

import data
import gen_encoder
import gen_decoder
from train import get_sent_lengths

import argparse
import pathlib
import os
import sys

import pandas as pd
import torch
import torch.nn.functional as F
import torchtext

CUSTOM_BATCH_SIZE = 5
MAX_GEN_LEN = 25
TOP_P = 0.8
TOP_K = 100

ABS_PATH = pathlib.Path(__file__).parent.absolute() 
RESULTS_PATH = os.path.join(str(ABS_PATH), 'results/')

MODEL_PATH_DICT = {
    "entailment": "models/entail-gen",
    "contradiction": "models/contra-gen",
}

MODEL_FNAMES = {
    "entailment":"entail-gen.tar",
    "contradiction":"contra-gen.tar"    
}

# Function to perform evaluation/write out results of model for a given batch
def test_batch(batch, encoder, decoder, rows_list, device, custom = False, use_topk = True):
    if custom:
        premises = batch
    else:
        premises = batch.premise

    premises.to(device)

    curr_batch_size = premises.size(0)
    
    result_dicts = [{} for _ in range(curr_batch_size)]

    for b in range(curr_batch_size):
        result_dicts[b]["premise"] = ""
        result_dicts[b]["hypothesis"] = "<sos> "

        for w in range(premises[b, :].size(0)):
            premise = premises[b]
            word_idx = premise[w]
            if word_idx != data.PAD_TOKEN_ID:
                result_dicts[b]["premise"] += data.inputs.vocab.itos[word_idx] + " "

    with torch.no_grad():
        # Feed input through encoder, store packed output + context
        encoder_out, (encoder_hidden, encoder_cell) = encoder(premises,
            get_sent_lengths(premises, device), encoder.initHidden(curr_batch_size, device), 
            encoder.initCell(curr_batch_size, device))

        # Decoder setup -> forward propogation
        decoder_hidden = encoder_hidden
        decoder_cell = encoder_cell

        decoder_input = torch.tensor([data.INIT_TOKEN_ID], device=device)
        decoder_input = decoder_input.repeat(curr_batch_size).view(curr_batch_size, 1)

        done_generating = [False for _ in range(curr_batch_size)]

        decoder_in_lengths = torch.ones(curr_batch_size, device=device)

        # Feed actual target token as input to next timestep
        for i in range(MAX_GEN_LEN):
            decoder_output, decoder_hidden, decoder_cell = decoder(decoder_input,
                decoder_in_lengths, decoder_hidden, decoder_cell, device)

            # squeeze seq. len (1)
            decoder_output = decoder_output.squeeze(1)

            # Input to next timestep are sampled from top-k/p of dist.
            decoder_input = torch.zeros_like(decoder_input)

            # Detokenize (to text) and write results to file
            for b in range(curr_batch_size):
                if done_generating[b]:
                    continue

                if use_topk:
                    topk_idxs = torch.topk(decoder_output[b], TOP_K)[1]
                    decoder_input[b] = torch.multinomial(F.softmax(decoder_output[b,topk_idxs], dim=-1), 1)

                else:
                    # use top-p sampling
                    sorted_logits, sorted_indices = torch.sort(decoder_output[b], descending = True)
                    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim = -1), dim  = -1)

                    idxs_to_rm = cum_probs > TOP_P

                    # shift right
                    idxs_to_rm[..., 1:] = idxs_to_rm[..., :-1].clone()
                    idxs_to_rm[..., 0] = 0

                    idxs_to_rm = sorted_indices[idxs_to_rm]

                    decoder_output[b,idxs_to_rm] = -float('Inf')

                    decoder_input[b] = torch.multinomial(F.softmax(decoder_output[b,:], dim=-1), 1)

                word_b = data.inputs.vocab.itos[decoder_input[b]]
                result_dicts[b]["hypothesis"] += word_b + " "

                if word_b == ".":
                    done_generating[b] = True

    rows_list.extend(result_dicts)
    

def main():
    device = data.device

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--contexts", type=str)

    args = parser.parse_args()

    if not os.path.isdir(RESULTS_PATH):
        print('creating directory %s ' % RESULTS_PATH)
        os.mkdir(RESULTS_PATH)
        os.mkdir(os.path.join(RESULTS_PATH,"entailment"))
        os.mkdir(os.path.join(RESULTS_PATH,"contradiction"))     

    print("Loading model")
    encoder = gen_encoder.Encoder(len(data.inputs.vocab), data.HIDDEN_SIZE,
        embeddings = data.inputs.vocab.vectors, embed_size = data.EMBED_SIZE,
        pad_idx = data.PAD_TOKEN_ID, unk_idx = data.UNK_TOKEN_ID)
    decoder = gen_decoder.Decoder(len(data.inputs.vocab), data.HIDDEN_SIZE, 
        embeddings = data.inputs.vocab.vectors, embed_size = data.EMBED_SIZE,
        pad_idx = data.PAD_TOKEN_ID, unk_idx = data.UNK_TOKEN_ID)

    MODEL_PATH = os.path.join(ABS_PATH, MODEL_PATH_DICT[args.model], MODEL_FNAMES[args.model])

    model = torch.load(MODEL_PATH, map_location=device)
    encoder.load_state_dict(model['encoder_state_dict'])
    decoder.load_state_dict(model['decoder_state_dict'])

    print("Starting Evaluation")

    encoder.eval()
    decoder.eval()

    encoder.to(device)
    decoder.to(device)

    with open(args.contexts, "r") as f:
        gen_contexts = f.read().splitlines() 

    #  convert contexts > tensor
    gen_contexts_tokenized = [[data.inputs.vocab.stoi[w] for w in context.split(' ')] for context in gen_contexts]

    # Use custom contexts on models
    print("Evaluating custom contexts")
    rows_custom = []

    # calc last batch
    num_custom_batches = int(len(gen_contexts) / CUSTOM_BATCH_SIZE)
    leftover = len(gen_contexts) % CUSTOM_BATCH_SIZE
    if leftover != 0:
        num_custom_batches += 1
        last_batch_size = leftover
    else:
        last_batch_size = CUSTOM_BATCH_SIZE

    for b in range(num_custom_batches):
        start_idx = b * CUSTOM_BATCH_SIZE

        if b == num_custom_batches - 1:
            curr_batch = [torch.tensor(context, device=device) for context in
                gen_contexts_tokenized[start_idx:start_idx + last_batch_size]]
        else:
            curr_batch = [torch.tensor(context, device=device) for context in
                gen_contexts_tokenized[start_idx:start_idx + CUSTOM_BATCH_SIZE]]
        
        curr_batch_padded = torch.nn.utils.rnn.pad_sequence(curr_batch, batch_first = True,
            padding_value = data.PAD_TOKEN_ID)
        
        test_batch(curr_batch_padded, encoder, decoder, rows_custom, device,
            custom = True, use_topk = False)

    df_custom = pd.DataFrame(rows_custom, columns = ("premise", "hypothesis"))
    df_custom.to_csv(os.path.join(RESULTS_PATH,args.model,"custom.csv"), sep = "\t",index = False)

    # Use test sets on models
    print("Evaluating test set")
    if args.model == "entailment":
        rows_list_entail = []
        for batch_num, batch in enumerate(data.test_iter_entail):
            test_batch(batch, encoder, decoder, rows_list_entail, device, custom = False, use_topk = False)

        df_entail = pd.DataFrame(rows_list_entail, columns = ("premise", "hypothesis"))
        df_entail.to_csv(os.path.join(RESULTS_PATH,args.model,"test.csv"), sep = "\t", index = False)
    elif args.model == "contradiction":
        rows_list_contradict = []
        for batch_num, batch in enumerate(data.test_iter_contradict):
            test_batch(batch, encoder, decoder, rows_list_contradict, device, custom = False, use_topk = False)
        df_contradict = pd.DataFrame(rows_list_contradict, columns = ("premise", "hypothesis"))
        df_contradict.to_csv(os.path.join(RESULTS_PATH,args.model,"test.csv"), sep = "\t", index = False)


if __name__ == "__main__":
    main()
