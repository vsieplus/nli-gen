# Evaluation script for the generation model

import data
import gen_encoder
import gen_decoder

import argparse
import pathlib
import os
import sys

import pandas as pd
import torch
import torch.nn.functional as F
import torchtext

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
            result_dicts[b]["premise"] += data.inputs.vocab.itos[word_idx] + " "

    with torch.no_grad():
        # Feed input through encoder, store packed output + context
        encoder_hidden = encoder.initHidden(curr_batch_size, device)
        encoder_cell = encoder.initCell(curr_batch_size, device)

        # Feed input through encoder
        encoder_out, (encoder_hidden, encoder_cell) = encoder(premises,
            encoder_hidden, encoder_cell)

        # Decoder setup -> forward propogation
        decoder_input = torch.tensor([[data.INIT_TOKEN_ID]], device=device)

        decoder_hidden = encoder_hidden
        decoder_cell = encoder_cell

        # Feed actual target token as input to next timestep
        for i in range(MAX_GEN_LEN):
            decoder_output, decoder_hidden, decoder_cell = decoder(decoder_input,
                decoder_hidden, decoder_cell, device)

            # Input to next timestep are sampled from top-k of dist.
            decoder.input=torch.zeros(curr_batch_size, device=device)

            # Detokenize (to text) and write results to file
            for b in range(curr_batch_size):
                if use_topk:
                    topk_idxs = torch.topk(decoder_output[b], TOP_K)[1]
                    decoder_input[b] = torch.multinomial(F.softmax(decoder_output[b,topk_idxs], dim=-1), 1)

                else:
                    # use top-p sampling
                    sorted_logits, sorted_indices = torch.sort(decoder_output[0], descending = True)
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
                    rows_list.extend(result_dicts)
                    return


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

    for b in range(len(gen_contexts)):
        curr_batch = torch.tensor(gen_contexts_tokenized[b], device=device)
        curr_batch = curr_batch.unsqueeze(0)
        test_batch(curr_batch, encoder, decoder, rows_custom, device, custom = True,
            use_topk = False)

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
            test_batch(batch, encoder, decoder, rows_list_contradict, device)
        df_contradict = pd.DataFrame(rows_list_contradict, columns = ("premise", "hypothesis"))
        df_contradict.to_csv(os.path.join(RESULTS_PATH,args.model,"test.csv"), sep = "\t", index = False)


if __name__ == "__main__":
    main()
