# Evaluation script for the generation model

import data
import gen_encoder
import gen_decoder

import argparse
import pathlib
import os

import torchtext


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
def test_batch(batch, encoder, decoder, rows_list):
    curr_batch_size = batch.batch_size
    result_dicts = [{} for _ in range(curr_batch_size)]

    for b in range(curr_batch_size):
        result_dicts[b]["premise"] = ""
        for w in range(batch.premise[b, :].size(1)):
            word_idx = batch.premise[b,w]
            result_dicts[b]["premise"] += data.inputs.iots[word_idx] + " "

    with torch.no_grad():
        # Feed input through encoder, store packed output + context
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

            curr_hidden = torch.where(not_padded_bool, encoder_hidden[0], torch.tensor(0.)).unsqueeze(0)
            curr_cell = torch.where(not_padded_bool, encoder_cell[0], torch.tensor(0.)).unsqueeze(0)

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
            if i > 0:
                decoder_input = hypotheses[:, i]
            else:
                decoder_input = decoder_input.unsqueeze(1)        

            decoder_output, decoder_hidden, decoder_cell, decoder_attn = decoder(decoder_input,
                curr_hidden, curr_cell, encoder_outputs, not_padded)

            # Input to next timestep are argmax indices of decoder output
            decoder_input = torch.argmax(decoder_output, dim = -1)

            # Detokenize (to text) and write results to file
            for b in range(curr_batch_size):
                result_dicts[b]["hypothesis"] = ""
                word_b = data.inputs.iots[decoder_input[b]]        
                result_dicts[b]["hypothesis"] += word_b + " "

            decoder_input = decoder_input.reshape(curr_batch_size, -1)

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

    gen_contexts_tokenized = [[data.inputs.stoi[w] for w in context] for context in gen_contexts]

    # Use custom contexts on models
    custom_batch = torchtext.data.Batch(gen_contexts_tokenized)
    rows_custom = []
    test_batch(custom_batch, encoder, decoder, rows_custom)
    df_custom = pd.DataFrame(rows_custom, columns = ("premise", "hypothesis"))
    df_custom.to_csv(os.path.join(RESULTS_PATH,args.model,"custom.csv"), sep = "\t")

    # Use train sets on models
    if args.model == "entailment":
        rows_list_entail = []
        for batch_num, batch in enumerate(data.test_iter_entail):
            test_batch(batch, encoder, decoder, rows_list_entail)
        df_entail = pd.DataFrame(rows_list_entail, columns = ("premise", "hypothesis"))
        df_entail.to_csv(os.path.join(RESULTS_PATH,args.model,"test.csv"), sep = "\t")
    elif args.model == "contradiction":
        rows_list_contradict = []
        for batch_num, batch in enumerate(data.test_iter_contradict):
            test_batch(batch, encoder, decoder, rows_list_contradict)
        df_contradict = pd.DataFrame(rows_list_contradict, columns = ("premise", "hypothesis"))
        df_contradict.to_csv(os.path.join(RESULTS_PATH,args.model,"test.csv"), sep = "\t")


if __name__ == "__main__":
    main()