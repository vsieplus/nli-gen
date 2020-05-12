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

INIT_TOKEN_ID = data.inputs.vocab.stoi[data.INIT_TOKEN]
PAD_ID = 1
MAX_GEN_LEN = 25

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
def test_batch(batch, encoder, decoder, rows_list, device, custom = False):
    if custom:
        premises = batch
    else:
        premises = batch.premise

    curr_batch_size = premises.size(0)
    
    result_dicts = [{} for _ in range(curr_batch_size)]

    for b in range(curr_batch_size):
        result_dicts[b]["premise"] = ""
        for w in range(premises[b, :].size(0)):
            premise = premises[b]
            word_idx = premise[w]
            result_dicts[b]["premise"] += data.inputs.vocab.itos[word_idx] + " "

    with torch.no_grad():
        # Feed input through encoder, store packed output + context
        encoder_hidden = encoder.initHidden(curr_batch_size, device)
        encoder_cell = encoder.initCell(curr_batch_size, device)
        encoder_outputs = torch.zeros(premises.size(0), premises.size(1), 
                            encoder.hidden_size, device=device)

        batch_idxs = torch.arange(curr_batch_size, dtype=torch.int64, device=device)
        seq_idxs = torch.arange(premises.size(1), dtype=torch.int64, device=device)

        # Feed input through encoder, track encoder outputs for attention
        for i in range(premises.size(1)):
            # Only pass examples not yet finished processing
            padded = [j for j in range(premises.size(0)) if premises[j,i] == PAD_ID]
            lengths = torch.ones(curr_batch_size, device=device)
            lengths[padded] = 0.0

#            not_padded = [j for j in range(premises.size(0)) if premises[j,i] != PAD_ID]
#            not_padded_bool = torch.tensor([idx in not_padded for idx in batch_idxs], 
#                                device=device)
#            not_padded_bool = not_padded_bool.view(-1, 1).repeat(1,
#                data.HIDDEN_SIZE).view(curr_batch_size, data.HIDDEN_SIZE)

            encoder_input = premises[:, i:i+1]

#            curr_hidden = torch.where(not_padded_bool, encoder_hidden[0], torch.tensor(0.)).unsqueeze(0)
#            curr_cell = torch.where(not_padded_bool, encoder_cell[0], torch.tensor(0.)).unsqueeze(0)

            encoder_out, (encoder_hidden, encoder_cell) = encoder(encoder_input,
                encoder_hidden, encoder_cell, lengths)

            encoder_outputs[:, i] = encoder_out[:, 0]

            # Update overall hidden/cell
#            encoder_hidden = torch.where(not_padded_bool, next_hidden[0], encoder_hidden[0])
#            encoder_cell = torch.where(not_padded_bool, next_cell[0], encoder_cell[0])

#            encoder_hidden = encoder_hidden.unsqueeze(0)
#            encoder_cell = encoder_cell.unsqueeze(0)

#            not_padded_and_seq_bool = torch.tensor([[b_idx in not_padded and seq_idx == i
#                for seq_idx in seq_idxs] for b_idx in batch_idxs], device=device)
#            not_padded_and_seq_bool = not_padded_and_seq_bool.unsqueeze(2).repeat(1,1,
#                data.HIDDEN_SIZE)

#            encoder_outputs = torch.where(not_padded_and_seq_bool, 
#                encoder_out[:,0:1].repeat(1,len(seq_idxs),1), encoder_outputs)

        # Decoder setup -> forward propogation
        decoder_input = torch.tensor([[INIT_TOKEN_ID]], device=device)
        decoder_input = decoder_input.repeat(curr_batch_size, 1)

        decoder_hidden = encoder_hidden
        decoder_cell = encoder_cell

        # Feed actual target token as input to next timestep
        for i in range(MAX_GEN_LEN):
            decoder_output, decoder_hidden, decoder_cell, decoder_attn = decoder(decoder_input,
                decoder_hidden, decoder_cell, encoder_outputs, torch.ones(curr_batch_size, 
                device=device), device)

            # Input to next timestep are sampled from top-k of dist.
            decoder.input=torch.zeros(curr_batch_size, device=device)

            # Detokenize (to text) and write results to file
            for b in range(curr_batch_size):
                topk_idxs = torch.topk(decoder_output[b], 1000)[1]
                decoder_input[b] = torch.multinomial(F.softmax(decoder_output[b,topk_idxs], dim=-1), 1)

                result_dicts[b]["hypothesis"] = ""
                word_b = data.inputs.vocab.itos[decoder_input[b]]        
                result_dicts[b]["hypothesis"] += word_b + " "


    rows_list.extend(result_dicts)
    

def main():
    device = torch.device("cpu")

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
        embeddings = data.inputs.vocab.vectors, embed_size = data.EMBED_SIZE)
    decoder = gen_decoder.Decoder(len(data.inputs.vocab), data.HIDDEN_SIZE, 
        embeddings = data.inputs.vocab.vectors, embed_size = data.EMBED_SIZE)

    encoder.to(device)
    decoder.to(device)

    MODEL_PATH = os.path.join(ABS_PATH, MODEL_PATH_DICT[args.model], MODEL_FNAMES[args.model])

    model = torch.load(MODEL_PATH, map_location=device)
    encoder.load_state_dict(model['encoder_state_dict'])
    decoder.load_state_dict(model['decoder_state_dict'])

    encoder.eval()
    decoder.eval()

    print("Starting Evaluation")

    encoder.eval()
    decoder.eval()

    with open(args.contexts, "r") as f:
        gen_contexts = f.read().splitlines() 

    # Pad contexts + convert > tensor
    gen_contexts_tokenized = [[data.inputs.vocab.stoi[w] for w in context] for context in gen_contexts]
    MAX_CONTEXT_LEN = max([len(tokens) for tokens in gen_contexts_tokenized])
    
    for k in range(len(gen_contexts_tokenized)):
        while len(gen_contexts_tokenized[k]) < MAX_CONTEXT_LEN:
            gen_contexts_tokenized[k].append(PAD_ID)        

    custom_batch = torch.tensor(gen_contexts_tokenized, device=device)

    # Use custom contexts on models
    print("Evaluating custom contexts")
    rows_custom = []
    test_batch(custom_batch, encoder, decoder, rows_custom, device, custom = True)
    df_custom = pd.DataFrame(rows_custom, columns = ("premise", "hypothesis"))
    df_custom.to_csv(os.path.join(RESULTS_PATH,args.model,"custom.csv"), sep = "\t",index = False)

    sys.exit()

    # Use train sets on models
    print("Evaluating test set")
    if args.model == "entailment":
        rows_list_entail = []
        for batch_num, batch in enumerate(data.test_iter_entail):
            test_batch(batch, encoder, decoder, rows_list_entail, device)
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
